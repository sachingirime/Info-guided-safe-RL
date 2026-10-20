// gaden_driver: persistent GADEN (gaden_core) process driven over stdin/stdout.
//
// Text command lines on stdin; binary payloads follow the line where noted.
// Replies are one text line ("OK ..." or "ERR ...") plus optional binary data.
//
//   INIT <env.bin> <params.txt>   build Environment + RunningSimulation (wind = 0)
//   WIND                          + nx*ny*nz*3 float32 (x fastest): replace current wind map
//   STEP <n>                      advance n timesteps  -> "OK <time> <n_filaments>"
//   QUERY <m>                     + m*3 float32 (m)    -> "OK" + m float32 (ppm)
//   FIL                           -> "OK <n>" + n*4 float32 (x, y, z [m], sigma [cm])
//   QUIT
//
// env.bin:    int32 nx, ny, nz; float32 minx, miny, minz, cellSize; uint8 cells[nx*ny*nz]
//             (x fastest, GADEN CellState: 0 free, 1 obstacle, 2 outlet)
// params.txt: "key value" lines (see ReadParams).  `seed` advances every OpenMP thread's
//             GADEN RNG engines before the first step (GADEN itself never seeds them, so all
//             threads otherwise share one identical noise table).
#include <gaden/RunningSimulation.hpp>
#include <gaden/internal/MathUtils.hpp>
#include <omp.h>

#include <cstdio>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>

using namespace gaden;

static void readExact(void* dst, size_t n)
{
    if (fread(dst, 1, n, stdin) != n)
        throw std::runtime_error("short read on stdin");
}

static void writeLine(const std::string& s)
{
    fwrite(s.data(), 1, s.size(), stdout);
    fputc('\n', stdout);
}

static std::map<std::string, std::string> ReadParams(const std::string& path)
{
    std::map<std::string, std::string> kv;
    std::ifstream in(path);
    if (!in)
        throw std::runtime_error("cannot open params " + path);
    std::string k, v;
    while (in >> k >> v)
        kv[k] = v;
    return kv;
}

static float getf(std::map<std::string, std::string>& kv, const char* k, float def)
{
    auto it = kv.find(k);
    return it == kv.end() ? def : std::stof(it->second);
}

int main()
{
    std::shared_ptr<EnvironmentConfiguration> cfg;
    std::unique_ptr<RunningSimulation> sim;
    size_t numCells = 0;
    std::string line;
    char* lbuf = nullptr;
    size_t lcap = 0;

    // C stdio only: binary payloads are fread from the same stdin buffer
    while (getline(&lbuf, &lcap, stdin) > 0)
    {
        line.assign(lbuf);
        while (!line.empty() && (line.back() == '\n' || line.back() == '\r'))
            line.pop_back();
        std::istringstream ls(line);
        std::string cmd;
        ls >> cmd;
        try
        {
            if (cmd == "INIT")
            {
                std::string envPath, parPath;
                ls >> envPath >> parPath;
                cfg = std::make_shared<EnvironmentConfiguration>();
                std::ifstream ef(envPath, std::ios::binary);
                if (!ef)
                    throw std::runtime_error("cannot open env " + envPath);
                int32_t dims[3];
                float hdr[4];
                ef.read((char*)dims, sizeof dims);
                ef.read((char*)hdr, sizeof hdr);
                auto& d = cfg->environment.description;
                d.dimensions = Vector3i(dims[0], dims[1], dims[2]);
                d.minCoord = Vector3(hdr[0], hdr[1], hdr[2]);
                d.cellSize = hdr[3];
                d.maxCoord = d.minCoord + Vector3(dims[0], dims[1], dims[2]) * d.cellSize;
                numCells = size_t(dims[0]) * dims[1] * dims[2];
                std::vector<uint8_t> raw(numCells);
                ef.read((char*)raw.data(), numCells);
                if (!ef)
                    throw std::runtime_error("env file truncated");
                cfg->environment.cells.resize(numCells);
                for (size_t i = 0; i < numCells; i++)
                    cfg->environment.cells[i] = static_cast<Environment::CellState>(raw[i]);
                cfg->windSequence.Initialize(std::vector<std::vector<Vector3>>{std::vector<Vector3>(numCells, Vector3(0))}, numCells, {});

                auto kv = ReadParams(parPath);
                RunningSimulation::Parameters p;
                auto src = std::make_shared<PointSource>();
                src->sourcePosition = Vector3(getf(kv, "source_x", 0), getf(kv, "source_y", 0), getf(kv, "source_z", 0));
                src->gasType = static_cast<GasType>(int(getf(kv, "gas_type", 0)));
                p.source = src;
                p.deltaTime = getf(kv, "deltaTime", 0.1f);
                p.temperature = getf(kv, "temperature", 298.f);
                p.pressure = getf(kv, "pressure", 1.f);
                p.filamentPPMcenter_initial = getf(kv, "filamentPPMcenter_initial", 20.f);
                p.filamentInitialSigma = getf(kv, "filamentInitialSigma", 10.f);
                p.filamentGrowthGamma = getf(kv, "filamentGrowthGamma", 10.f);
                p.filamentNoise_std = getf(kv, "filamentNoise_std", 0.02f);
                p.numFilaments_sec = getf(kv, "numFilaments_sec", 10.f);
                p.expectedNumIterations = size_t(getf(kv, "expectedNumIterations", 600));
                p.saveResults = false;

                // Seed: advance each thread's engines (inline-function statics are shared with
                // libgaden), before RunningSimulation.cpp builds its per-thread Gaussian table.
                long seed = long(getf(kv, "seed", 0));
                if (seed > 0)
                {
#pragma omp parallel
                    {
                        long n = seed * 7919L + omp_get_thread_num() * 104729L;
                        for (long i = 0; i < n; i++)
                        {
                            GaussianRandom(0, 1);
                            uniformRandom(0, 1);
                        }
                    }
                }
                sim = std::make_unique<RunningSimulation>(p, cfg);
                writeLine("OK " + std::to_string(numCells));
            }
            else if (cmd == "WIND")
            {
                if (!cfg)
                    throw std::runtime_error("INIT first");
                auto& w = cfg->windSequence.GetCurrent();
                readExact(w.data(), numCells * sizeof(Vector3));
                writeLine("OK");
            }
            else if (cmd == "STEP")
            {
                long n = 1;
                ls >> n;
                for (long i = 0; i < n; i++)
                    sim->AdvanceTimestep();
                char buf[96];
                snprintf(buf, sizeof buf, "OK %.6f %zu", sim->GetCurrentTime(), sim->GetFilaments().size());
                writeLine(buf);
            }
            else if (cmd == "QUERY")
            {
                long m = 0;
                ls >> m;
                std::vector<Vector3> pts(m);
                readExact(pts.data(), m * sizeof(Vector3));
                std::vector<float> out(m);
                auto const& env = cfg->environment;
#pragma omp parallel for schedule(dynamic, 64)
                for (long i = 0; i < m; i++)
                    out[i] = env.IsInBounds(pts[i]) ? sim->SampleConcentration(pts[i]) : 0.f;
                writeLine("OK");
                fwrite(out.data(), sizeof(float), m, stdout);
            }
            else if (cmd == "FIL")
            {
                auto const& f = sim->GetFilaments();
                writeLine("OK " + std::to_string(f.size()));
                std::vector<float> buf(f.size() * 4);
                for (size_t i = 0; i < f.size(); i++)
                {
                    buf[4 * i] = f[i].position.x;
                    buf[4 * i + 1] = f[i].position.y;
                    buf[4 * i + 2] = f[i].position.z;
                    buf[4 * i + 3] = f[i].sigma;
                }
                fwrite(buf.data(), sizeof(float), buf.size(), stdout);
            }
            else if (cmd == "QUIT")
            {
                writeLine("OK");
                fflush(stdout);
                return 0;
            }
            else
                writeLine("ERR unknown command '" + cmd + "'");
        }
        catch (std::exception& e)
        {
            writeLine(std::string("ERR ") + e.what());
        }
        fflush(stdout);
    }
    return 0;
}
