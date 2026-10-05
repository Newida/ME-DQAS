# DQAS vs ME-DQAS MaxCut Dataset

These instances were selected to be neither flat nor immediately solved under the saved hyperparameters.
The MaxCut objective is minimized as the number of uncut edges, so lower values are better.

## Default Run Settings

- n: 8
- circuit length L: 8
- gate set: rx ry rz cz
- batch size: 128
- S_f: 50
- S_theta: 50
- budget: 750000

## Problems

| problem | alpha | seed | edges | DQAS updates | ME updates | DQAS final | ME final | ME theta ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| problem_001_search_alpha1.5_seed20260750 | 1.5 | 20260750 | 12 | 9 | 14 | 10.98 | 9.878 | 0.6092 |
| problem_002_search_alpha2_seed20260750 | 2 | 20260750 | 16 | 9 | 13 | 13.49 | 10.53 | 0.6178 |
| problem_003_search_alpha1.5_seed20260751 | 1.5 | 20260751 | 12 | 9 | 13 | 10.22 | 8.423 | 0.6109 |
| problem_004_search_alpha2_seed20260751 | 2 | 20260751 | 16 | 9 | 13 | 12.96 | 9.745 | 0.6258 |
| problem_005_search_alpha1.5_seed20260752 | 1.5 | 20260752 | 12 | 9 | 15 | 9.982 | 7.937 | 0.5442 |
| problem_006_search_alpha2_seed20260752 | 2 | 20260752 | 16 | 9 | 13 | 13.48 | 11.86 | 0.6199 |
| problem_007_search_alpha1.5_seed20260753 | 1.5 | 20260753 | 12 | 9 | 14 | 10.64 | 8.916 | 0.6038 |
| problem_008_search_alpha2_seed20260753 | 2 | 20260753 | 16 | 9 | 13 | 14.07 | 12.45 | 0.6094 |
| problem_009_search_alpha1.5_seed20260754 | 1.5 | 20260754 | 12 | 8 | 13 | 10.1 | 8.14 | 0.6195 |
| problem_010_search_alpha2_seed20260754 | 2 | 20260754 | 16 | 8 | 13 | 12.83 | 9.821 | 0.6312 |
| problem_011_search_alpha1.5_seed20260755 | 1.5 | 20260755 | 12 | 9 | 16 | 11.18 | 10.25 | 0.5212 |
| problem_012_search_alpha2_seed20260755 | 2 | 20260755 | 16 | 9 | 13 | 12.99 | 10.42 | 0.6134 |
| problem_013_search_alpha1.5_seed20260756 | 1.5 | 20260756 | 12 | 9 | 14 | 10.82 | 9.282 | 0.6016 |
| problem_014_search_alpha2_seed20260756 | 2 | 20260756 | 16 | 9 | 14 | 14.34 | 12.7 | 0.6066 |
| problem_015_search_alpha1.5_seed20260757 | 1.5 | 20260757 | 12 | 9 | 14 | 11.11 | 10.35 | 0.6004 |
| problem_016_search_alpha2_seed20260757 | 2 | 20260757 | 16 | 9 | 13 | 12.55 | 10.7 | 0.6197 |
| problem_017_search_alpha1.5_seed20260758 | 1.5 | 20260758 | 12 | 9 | 13 | 9.256 | 7.674 | 0.6245 |
| problem_018_search_alpha2_seed20260758 | 2 | 20260758 | 16 | 9 | 13 | 12.77 | 10.75 | 0.6328 |
| problem_019_search_alpha1.5_seed20260759 | 1.5 | 20260759 | 12 | 8 | 13 | 9.873 | 6.866 | 0.6183 |
| problem_020_search_alpha2_seed20260759 | 2 | 20260759 | 16 | 8 | 13 | 13.17 | 9.897 | 0.6276 |
| problem_021_search_alpha1.5_seed20260760 | 1.5 | 20260760 | 12 | 9 | 14 | 10.85 | 10.01 | 0.6094 |
| problem_022_search_alpha2_seed20260760 | 2 | 20260760 | 16 | 9 | 13 | 10.44 | 8.78 | 0.635 |
| problem_023_search_alpha1.5_seed20260761 | 1.5 | 20260761 | 12 | 9 | 14 | 10.86 | 9.606 | 0.6068 |
| problem_024_search_alpha2_seed20260761 | 2 | 20260761 | 16 | 8 | 13 | 13.75 | 11.15 | 0.6222 |
| problem_025_search_alpha1.5_seed20260762 | 1.5 | 20260762 | 12 | 9 | 13 | 10.18 | 9.365 | 0.6184 |
| problem_026_search_alpha2_seed20260762 | 2 | 20260762 | 16 | 8 | 13 | 13.27 | 10.46 | 0.635 |
| problem_027_search_alpha1.5_seed20260763 | 1.5 | 20260763 | 12 | 8 | 14 | 10.87 | 9.338 | 0.6051 |
| problem_028_search_alpha2_seed20260763 | 2 | 20260763 | 16 | 8 | 13 | 14.36 | 12.32 | 0.6142 |
| problem_029_search_alpha1.5_seed20260764 | 1.5 | 20260764 | 12 | 9 | 14 | 11.65 | 11.15 | 0.5874 |
| problem_030_search_alpha2_seed20260764 | 2 | 20260764 | 16 | 9 | 14 | 15.81 | 13.87 | 0.5933 |
| problem_031_search_alpha1.5_seed20260765 | 1.5 | 20260765 | 12 | 8 | 13 | 10.32 | 8.475 | 0.6146 |
| problem_032_search_alpha2_seed20260765 | 2 | 20260765 | 16 | 8 | 13 | 13.29 | 10.52 | 0.6241 |
| problem_033_search_alpha1.5_seed20260766 | 1.5 | 20260766 | 12 | 8 | 13 | 10.9 | 9.511 | 0.6078 |
| problem_034_search_alpha2_seed20260766 | 2 | 20260766 | 16 | 8 | 13 | 14.36 | 11.73 | 0.6152 |
| problem_035_search_alpha1.5_seed20260767 | 1.5 | 20260767 | 12 | 9 | 13 | 10 | 8.89 | 0.61 |
| problem_036_search_alpha2_seed20260767 | 2 | 20260767 | 16 | 9 | 13 | 14.11 | 11.95 | 0.6133 |
| problem_037_search_alpha1.5_seed20260768 | 1.5 | 20260768 | 12 | 8 | 13 | 10.11 | 8.607 | 0.6102 |
| problem_038_search_alpha2_seed20260768 | 2 | 20260768 | 16 | 8 | 13 | 12.77 | 8.987 | 0.6282 |
| problem_039_search_alpha1.5_seed20260769 | 1.5 | 20260769 | 12 | 9 | 14 | 10.92 | 9.804 | 0.6087 |
| problem_040_search_alpha2_seed20260769 | 2 | 20260769 | 16 | 8 | 13 | 11.6 | 8.999 | 0.6309 |
| problem_041_search_alpha1.5_seed20260770 | 1.5 | 20260770 | 12 | 8 | 14 | 11.22 | 10.55 | 0.5998 |
| problem_042_search_alpha2_seed20260770 | 2 | 20260770 | 16 | 8 | 13 | 13.17 | 10.13 | 0.6206 |
| problem_043_search_alpha1.5_seed20260771 | 1.5 | 20260771 | 12 | 8 | 14 | 10.88 | 9.966 | 0.6045 |
| problem_044_search_alpha2_seed20260771 | 2 | 20260771 | 16 | 8 | 13 | 13.06 | 10.72 | 0.6178 |
| problem_045_search_alpha1.5_seed20260772 | 1.5 | 20260772 | 12 | 9 | 14 | 10.67 | 8.864 | 0.5957 |
| problem_046_search_alpha2_seed20260772 | 2 | 20260772 | 16 | 9 | 14 | 14.44 | 12.43 | 0.598 |
| problem_047_search_alpha1.5_seed20260773 | 1.5 | 20260773 | 12 | 8 | 13 | 10.25 | 8.295 | 0.6173 |
| problem_048_search_alpha2_seed20260773 | 2 | 20260773 | 16 | 8 | 13 | 13.58 | 11.24 | 0.6298 |
| problem_049_search_alpha1.5_seed20260774 | 1.5 | 20260774 | 12 | 9 | 15 | 10.47 | 8.862 | 0.5239 |
| problem_050_search_alpha2_seed20260774 | 2 | 20260774 | 16 | 8 | 13 | 13.66 | 11.21 | 0.6116 |
| problem_051_search_alpha1.5_seed20260775 | 1.5 | 20260775 | 12 | 9 | 14 | 10.6 | 9.225 | 0.6074 |
| problem_052_search_alpha2_seed20260775 | 2 | 20260775 | 16 | 9 | 13 | 14.07 | 11.31 | 0.6166 |
| problem_053_search_alpha1.5_seed20260776 | 1.5 | 20260776 | 12 | 9 | 16 | 10.76 | 8.605 | 0.5205 |
| problem_054_search_alpha2_seed20260776 | 2 | 20260776 | 16 | 9 | 14 | 13.81 | 10.76 | 0.6091 |
| problem_055_search_alpha1.5_seed20260777 | 1.5 | 20260777 | 12 | 8 | 13 | 10.23 | 8.186 | 0.6132 |
| problem_056_search_alpha2_seed20260777 | 2 | 20260777 | 16 | 8 | 13 | 13.84 | 11.86 | 0.6149 |
| problem_057_search_alpha1.5_seed20260778 | 1.5 | 20260778 | 12 | 9 | 14 | 11.02 | 9.9 | 0.6055 |
| problem_058_search_alpha2_seed20260778 | 2 | 20260778 | 16 | 9 | 13 | 14.02 | 12.59 | 0.6126 |
| problem_059_search_alpha1.5_seed20260779 | 1.5 | 20260779 | 12 | 9 | 14 | 10.49 | 9.48 | 0.6055 |
| problem_060_search_alpha2_seed20260779 | 2 | 20260779 | 16 | 8 | 13 | 13.33 | 9.356 | 0.6244 |
| problem_061_search_alpha1.5_seed20260780 | 1.5 | 20260780 | 12 | 9 | 14 | 8.272 | 7.115 | 0.5937 |
| problem_062_search_alpha2_seed20260780 | 2 | 20260780 | 16 | 8 | 14 | 11.05 | 9.119 | 0.6059 |
| problem_063_search_alpha1.5_seed20260781 | 1.5 | 20260781 | 12 | 9 | 14 | 11.06 | 10.35 | 0.6011 |
| problem_064_search_alpha2_seed20260781 | 2 | 20260781 | 16 | 9 | 14 | 14.65 | 12.15 | 0.6069 |
| problem_065_search_alpha1.5_seed20260782 | 1.5 | 20260782 | 12 | 9 | 14 | 11.25 | 10.38 | 0.6021 |
| problem_066_search_alpha2_seed20260782 | 2 | 20260782 | 16 | 9 | 14 | 14.65 | 12.3 | 0.6107 |
| problem_067_search_alpha1.5_seed20260783 | 1.5 | 20260783 | 12 | 9 | 14 | 10.89 | 9.918 | 0.5991 |
| problem_068_search_alpha2_seed20260783 | 2 | 20260783 | 16 | 8 | 13 | 12.78 | 9.935 | 0.6125 |
| problem_069_search_alpha1.5_seed20260784 | 1.5 | 20260784 | 12 | 8 | 13 | 10.27 | 8.722 | 0.6015 |
| problem_070_search_alpha2_seed20260784 | 2 | 20260784 | 16 | 8 | 13 | 11.71 | 8.545 | 0.6134 |
| problem_071_search_alpha1.5_seed20260785 | 1.5 | 20260785 | 12 | 9 | 14 | 10.26 | 8.245 | 0.6064 |
| problem_072_search_alpha2_seed20260785 | 2 | 20260785 | 16 | 9 | 13 | 13.08 | 10.02 | 0.6209 |
| problem_073_search_alpha1.5_seed20260786 | 1.5 | 20260786 | 12 | 9 | 15 | 9.558 | 6.861 | 0.5375 |
| problem_074_search_alpha2_seed20260786 | 2 | 20260786 | 16 | 9 | 13 | 13.42 | 11.28 | 0.6226 |
| problem_075_search_alpha1.5_seed20260787 | 1.5 | 20260787 | 12 | 8 | 14 | 11.27 | 10.46 | 0.599 |
| problem_076_search_alpha2_seed20260787 | 2 | 20260787 | 16 | 8 | 13 | 14.3 | 12.47 | 0.609 |
| problem_077_search_alpha1.5_seed20260788 | 1.5 | 20260788 | 12 | 9 | 14 | 11.06 | 10.14 | 0.6028 |
| problem_078_search_alpha2_seed20260788 | 2 | 20260788 | 16 | 9 | 13 | 14.1 | 11.34 | 0.615 |
| problem_079_search_alpha1.5_seed20260789 | 1.5 | 20260789 | 12 | 9 | 15 | 10.15 | 6.828 | 0.5341 |
| problem_080_search_alpha2_seed20260789 | 2 | 20260789 | 16 | 9 | 13 | 11.37 | 9.26 | 0.6267 |
| problem_081_search_alpha1.5_seed20260790 | 1.5 | 20260790 | 12 | 8 | 14 | 11.11 | 8.971 | 0.6028 |
| problem_082_search_alpha2_seed20260790 | 2 | 20260790 | 16 | 8 | 14 | 14.78 | 12.25 | 0.607 |
| problem_083_search_alpha1.5_seed20260791 | 1.5 | 20260791 | 12 | 9 | 14 | 10.72 | 9.174 | 0.6044 |
| problem_084_search_alpha2_seed20260791 | 2 | 20260791 | 16 | 8 | 13 | 14.31 | 11.79 | 0.6172 |
| problem_085_search_alpha1.5_seed20260792 | 1.5 | 20260792 | 12 | 9 | 15 | 10.57 | 8.741 | 0.5293 |
| problem_086_search_alpha2_seed20260792 | 2 | 20260792 | 16 | 9 | 13 | 13.37 | 10.77 | 0.612 |
| problem_087_search_alpha1.5_seed20260793 | 1.5 | 20260793 | 12 | 8 | 13 | 9.957 | 7.852 | 0.6163 |
| problem_088_search_alpha2_seed20260793 | 2 | 20260793 | 16 | 8 | 13 | 13.22 | 10.43 | 0.6267 |
| problem_089_search_alpha1.5_seed20260794 | 1.5 | 20260794 | 12 | 8 | 14 | 10.99 | 9.646 | 0.598 |
| problem_090_search_alpha2_seed20260794 | 2 | 20260794 | 16 | 8 | 14 | 14.51 | 11.98 | 0.6041 |
| problem_091_search_alpha1.5_seed20260795 | 1.5 | 20260795 | 12 | 8 | 14 | 11.21 | 10.29 | 0.5984 |
| problem_092_search_alpha2_seed20260795 | 2 | 20260795 | 16 | 8 | 13 | 14.53 | 12.76 | 0.6084 |
| problem_093_search_alpha1.5_seed20260796 | 1.5 | 20260796 | 12 | 8 | 13 | 10.38 | 8.75 | 0.609 |
| problem_094_search_alpha2_seed20260796 | 2 | 20260796 | 16 | 8 | 13 | 12.91 | 9.907 | 0.6249 |
| problem_095_search_alpha1.5_seed20260797 | 1.5 | 20260797 | 12 | 9 | 14 | 9.514 | 8.026 | 0.6086 |
| problem_096_search_alpha2_seed20260797 | 2 | 20260797 | 16 | 8 | 13 | 14.13 | 11.5 | 0.6131 |
| problem_097_search_alpha1.5_seed20260798 | 1.5 | 20260798 | 12 | 9 | 14 | 10.59 | 8.299 | 0.6028 |
| problem_098_search_alpha2_seed20260798 | 2 | 20260798 | 16 | 8 | 13 | 13.32 | 9.825 | 0.6172 |
| problem_099_search_alpha1.5_seed20260799 | 1.5 | 20260799 | 12 | 9 | 14 | 11.01 | 9.624 | 0.6102 |
| problem_100_search_alpha2_seed20260799 | 2 | 20260799 | 16 | 9 | 13 | 14.23 | 12.47 | 0.6156 |

Each problem directory contains:

- `problem.json`: graph edges, seeds, exact brute-force stats, and selection metrics
- `comparison.json`: full paired DQAS and ME-DQAS run logs
- `summary.csv`: one row per algorithm
- `loss_vs_updates.png`, `loss_vs_shots.png`, `batch_min_mean_max.png`, `final_metrics.png`

The comparison is stochastic; use the saved seeds for reproducibility.
