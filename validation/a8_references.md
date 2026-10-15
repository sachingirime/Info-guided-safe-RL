# A8: tex hygiene (wind_field_method.tex, CDC26 repo)

- Only `wind_field_method.tex` is staged in `~/CDC26/CDC26-Info-guided-safe-RL`; other uncommitted
  changes there (figures/sim_*, puff_concentration_slice.png, training_viz.py) are untouched.
  `figures/validation/` (the six validation figures the tex includes) is untracked and not staged.
- Compiles with Tectonic 0.17.0 (conda env `tex`): no errors, no overfull boxes.
- V3/V4 numbers in the tex are pre-A5 and are refreshed from the Part C validation run.

## Reference check (Crossref API / publisher pages, 2026-10-14)

| Reference | Verified | Source |
|---|---|---|
| Stam 1999, "Stable fluids", SIGGRAPH '99, pp. 121–128 | yes, doi:10.1145/311535.311548 | Crossref |
| Stauffer & Seaman 1990, Mon. Wea. Rev. 118(6), 1250–1277 | yes, doi:10.1175/1520-0493(1990)118<1250:UOFDDA>2.0.CO;2 | Crossref |
| Anfossi, Rizza, Mangia, Degrazia, Pereira Marques Filho 2006, Atmos. Environ. 40, 326–337 | yes, doi:10.1016/j.atmosenv.2005.09.041 (author list and title corrected in the tex) | Crossref |
| Gifford 1959, Adv. Geophys. 6, 117–137 | yes, doi:10.1016/S0065-2687(08)60099-0 | Crossref |
| Taylor, "Diffusion by continuous movements", Proc. London Math. Soc. s2-20, 196–212 | yes, doi:10.1112/plms/s2-20.1.196 (volume dated 1921–22) | Crossref |
| Thomson 1987, J. Fluid Mech. 180, 529–556 | yes, doi:10.1017/S0022112087001940 | Crossref |
| Briggs 1973, "Diffusion estimation for small emissions", ATDL Oak Ridge, preliminary draft | yes (dated 05/1973 in the UNT digital library record) | digital.library.unt.edu ark:/67531/metadc1057441 |
| Monroy et al. 2017 (GADEN), Sensors 17(7), 1479 | yes, doi:10.3390/s17071479 | Crossref |
| Ojeda, Monroy, Gonzalez-Jimenez 2025 (Gaden-RT), SoftwareX 32, 102388 | yes, doi:10.1016/j.softx.2025.102388 (not cited in the tex) | MAPIR/UMA records |
| Farrell et al. 2002, Environ. Fluid Mech. 2, 143–169 | DTIC report version verified (doi:10.21236/ADA399832); journal volume/pages from the search snippet only | Crossref |
| Young 1971, Iterative Solution of Large Linear Systems, Academic Press | yes (Elsevier/Academic Press, ISBN 978-0-12-773050-9) | Crossref |
| Stull 1988, Kluwer | book exists; the page numbers 454–455 used before could not be verified and were removed | — |
| NOAA READY PG class page | URL live (HTTP 200) | — |

Not verified: Hay & Pasquill (1959) bibliographic details (not cited; the tex names the hypothesis and cites Anfossi et al. 2006); Farrell et al. 2002 journal pagination (cited as in the earlier search result).
