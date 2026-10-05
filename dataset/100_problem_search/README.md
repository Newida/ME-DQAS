# DQAS vs ME-DQAS 3-SAT Dataset

These instances were selected to be neither flat nor immediately solved under the saved hyperparameters.
Each problem uses a high architecture batch size so the ME-DQAS shot-budget advantage is visible.

## Default Run Settings

- n: 8
- circuit length L: 8
- gate set: rx ry rz cz
- batch size: 128
- S_f: 50
- S_theta: 50
- budget: 750000

## Problems

| problem | alpha | seed | DQAS updates | ME updates | DQAS final | ME final | ME theta ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| problem_001_search_alpha1.5_seed20260627 | 1.5 | 20260627 | 9 | 14 | 3.477 | 3.333 | 0.6053 |
| problem_002_search_alpha2_seed20260627 | 2 | 20260627 | 9 | 14 | 4.723 | 4.02 | 0.6146 |
| problem_003_search_alpha1.5_seed20260628 | 1.5 | 20260628 | 9 | 14 | 1.767 | 1.701 | 0.596 |
| problem_004_search_alpha2_seed20260628 | 2 | 20260628 | 9 | 14 | 1.901 | 1.88 | 0.5924 |
| problem_005_search_alpha1.5_seed20260629 | 1.5 | 20260629 | 9 | 14 | 0.9798 | 0.9628 | 0.5894 |
| problem_006_search_alpha2_seed20260629 | 2 | 20260629 | 9 | 14 | 1.864 | 1.79 | 0.5919 |
| problem_007_search_alpha1.5_seed20260630 | 1.5 | 20260630 | 9 | 14 | 0.9778 | 0.967 | 0.5891 |
| problem_008_search_alpha2_seed20260630 | 2 | 20260630 | 9 | 14 | 0.9781 | 0.9675 | 0.589 |
| problem_009_search_alpha1.5_seed20260631 | 1.5 | 20260631 | 9 | 14 | 2.752 | 2.686 | 0.5897 |
| problem_010_search_alpha2_seed20260631 | 2 | 20260631 | 9 | 14 | 2.753 | 2.688 | 0.5897 |
| problem_011_search_alpha1.5_seed20260632 | 1.5 | 20260632 | 9 | 14 | 1.032 | 1.025 | 0.5832 |
| problem_012_search_alpha2_seed20260632 | 2 | 20260632 | 9 | 14 | 1.071 | 1.057 | 0.5815 |
| problem_013_search_alpha2_seed20260635 | 2 | 20260635 | 9 | 14 | 1.996 | 1.985 | 0.5808 |
| problem_014_search_alpha1.5_seed20260636 | 1.5 | 20260636 | 9 | 14 | 3.017 | 2.992 | 0.5848 |
| problem_015_search_alpha2_seed20260636 | 2 | 20260636 | 9 | 14 | 3.026 | 3 | 0.5842 |
| problem_016_search_alpha2_seed20260639 | 2 | 20260639 | 9 | 14 | 1.096 | 1.066 | 0.5786 |
| problem_017_search_alpha2_seed20260640 | 2 | 20260640 | 9 | 14 | 3.915 | 3.857 | 0.5828 |
| problem_018_search_alpha1.5_seed20260641 | 1.5 | 20260641 | 9 | 14 | 1.969 | 1.874 | 0.5849 |
| problem_019_search_alpha2_seed20260641 | 2 | 20260641 | 9 | 14 | 2.884 | 2.775 | 0.5875 |
| problem_020_search_alpha2_seed20260642 | 2 | 20260642 | 9 | 14 | 1.025 | 1.01 | 0.5794 |
| problem_021_search_alpha1.5_seed20260643 | 1.5 | 20260643 | 9 | 14 | 0.8416 | 0.8266 | 0.5768 |
| problem_022_search_alpha2_seed20260643 | 2 | 20260643 | 9 | 14 | 0.842 | 0.8256 | 0.5769 |
| problem_023_search_alpha1.5_seed20260644 | 1.5 | 20260644 | 9 | 14 | 1.93 | 1.88 | 0.5894 |
| problem_024_search_alpha2_seed20260644 | 2 | 20260644 | 9 | 14 | 2.856 | 2.775 | 0.5915 |
| problem_025_search_alpha1.5_seed20260650 | 1.5 | 20260650 | 9 | 14 | 2.894 | 2.881 | 0.5854 |
| problem_026_search_alpha2_seed20260650 | 2 | 20260650 | 9 | 14 | 3.777 | 3.72 | 0.5877 |
| problem_027_search_alpha1.5_seed20260651 | 1.5 | 20260651 | 9 | 14 | 1.947 | 1.933 | 0.5894 |
| problem_028_search_alpha2_seed20260651 | 2 | 20260651 | 9 | 14 | 3.712 | 3.622 | 0.5953 |
| problem_029_search_alpha1.5_seed20260653 | 1.5 | 20260653 | 9 | 14 | 2.939 | 2.901 | 0.5839 |
| problem_030_search_alpha2_seed20260653 | 2 | 20260653 | 9 | 14 | 4.836 | 4.689 | 0.5865 |
| problem_031_search_alpha1.5_seed20260655 | 1.5 | 20260655 | 9 | 14 | 1.874 | 1.8 | 0.5798 |
| problem_032_search_alpha2_seed20260655 | 2 | 20260655 | 9 | 14 | 2.804 | 2.699 | 0.5817 |
| problem_033_search_alpha2_seed20260656 | 2 | 20260656 | 9 | 14 | 1.969 | 1.946 | 0.5867 |
| problem_034_search_alpha1.5_seed20260657 | 1.5 | 20260657 | 9 | 14 | 1.039 | 1.008 | 0.5831 |
| problem_035_search_alpha2_seed20260657 | 2 | 20260657 | 9 | 14 | 1.993 | 1.935 | 0.584 |
| problem_036_search_alpha1.5_seed20260660 | 1.5 | 20260660 | 9 | 14 | 2.78 | 2.69 | 0.5883 |
| problem_037_search_alpha2_seed20260660 | 2 | 20260660 | 9 | 14 | 3.664 | 3.391 | 0.5894 |
| problem_038_search_alpha1.5_seed20260661 | 1.5 | 20260661 | 9 | 14 | 1.917 | 1.892 | 0.5845 |
| problem_039_search_alpha2_seed20260661 | 2 | 20260661 | 9 | 14 | 2.77 | 2.649 | 0.5873 |
| problem_040_search_alpha1.5_seed20260662 | 1.5 | 20260662 | 9 | 14 | 2.887 | 2.804 | 0.5806 |
| problem_041_search_alpha2_seed20260662 | 2 | 20260662 | 9 | 14 | 2.89 | 2.808 | 0.5804 |
| problem_042_search_alpha1.5_seed20260663 | 1.5 | 20260663 | 9 | 14 | 1.96 | 1.932 | 0.5801 |
| problem_043_search_alpha2_seed20260663 | 2 | 20260663 | 9 | 15 | 2.048 | 2.028 | 0.5776 |
| problem_044_search_alpha1.5_seed20260665 | 1.5 | 20260665 | 9 | 14 | 4.179 | 3.626 | 0.5985 |
| problem_045_search_alpha2_seed20260665 | 2 | 20260665 | 9 | 14 | 4.871 | 3.673 | 0.6027 |
| problem_046_search_alpha1.5_seed20260666 | 1.5 | 20260666 | 9 | 14 | 0.9538 | 0.9178 | 0.5896 |
| problem_047_search_alpha2_seed20260666 | 2 | 20260666 | 9 | 14 | 0.9542 | 0.9195 | 0.5895 |
| problem_048_search_alpha1.5_seed20260667 | 1.5 | 20260667 | 9 | 14 | 1.009 | 0.9914 | 0.5786 |
| problem_049_search_alpha2_seed20260667 | 2 | 20260667 | 9 | 14 | 1.952 | 1.945 | 0.5798 |
| problem_050_search_alpha1.5_seed20260668 | 1.5 | 20260668 | 9 | 14 | 1.96 | 1.938 | 0.5849 |
| problem_051_search_alpha2_seed20260668 | 2 | 20260668 | 9 | 14 | 1.987 | 1.959 | 0.5838 |
| problem_052_search_alpha2_seed20260670 | 2 | 20260670 | 9 | 14 | 1.013 | 0.9856 | 0.5811 |
| problem_053_search_alpha2_seed20260673 | 2 | 20260673 | 9 | 14 | 1.008 | 1.015 | 0.5798 |
| problem_054_search_alpha1.5_seed20260674 | 1.5 | 20260674 | 9 | 14 | 1.06 | 1.039 | 0.5824 |
| problem_055_search_alpha2_seed20260674 | 2 | 20260674 | 9 | 14 | 2.027 | 1.994 | 0.583 |
| problem_056_search_alpha1.5_seed20260675 | 1.5 | 20260675 | 9 | 14 | 0.945 | 0.9264 | 0.5796 |
| problem_057_search_alpha2_seed20260675 | 2 | 20260675 | 9 | 15 | 0.9923 | 0.9723 | 0.5758 |
| problem_058_search_alpha1.5_seed20260676 | 1.5 | 20260676 | 9 | 14 | 1.872 | 1.853 | 0.589 |
| problem_059_search_alpha2_seed20260676 | 2 | 20260676 | 9 | 14 | 1.88 | 1.867 | 0.589 |
| problem_060_search_alpha1.5_seed20260677 | 1.5 | 20260677 | 9 | 14 | 1.031 | 1.02 | 0.5809 |
| problem_061_search_alpha2_seed20260677 | 2 | 20260677 | 9 | 14 | 1.031 | 1.02 | 0.5809 |
| problem_062_search_alpha1.5_seed20260678 | 1.5 | 20260678 | 9 | 14 | 1.909 | 1.856 | 0.5917 |
| problem_063_search_alpha2_seed20260678 | 2 | 20260678 | 9 | 14 | 3.74 | 3.609 | 0.5931 |
| problem_064_search_alpha1.5_seed20260679 | 1.5 | 20260679 | 9 | 14 | 0.8789 | 0.893 | 0.5884 |
| problem_065_search_alpha2_seed20260679 | 2 | 20260679 | 9 | 14 | 0.9138 | 0.93 | 0.5872 |
| problem_066_search_alpha2_seed20260680 | 2 | 20260680 | 9 | 14 | 0.9866 | 0.9686 | 0.5797 |
| problem_067_search_alpha2_seed20260681 | 2 | 20260681 | 9 | 14 | 2.052 | 2.032 | 0.5799 |
| problem_068_search_alpha1.5_seed20260683 | 1.5 | 20260683 | 9 | 14 | 1.869 | 1.688 | 0.58 |
| problem_069_search_alpha2_seed20260683 | 2 | 20260683 | 9 | 14 | 2.831 | 2.652 | 0.5808 |
| problem_070_search_alpha1.5_seed20260684 | 1.5 | 20260684 | 9 | 14 | 0.9903 | 0.9953 | 0.582 |
| problem_071_search_alpha2_seed20260684 | 2 | 20260684 | 9 | 14 | 1.88 | 1.878 | 0.5848 |
| problem_072_search_alpha1.5_seed20260685 | 1.5 | 20260685 | 9 | 14 | 0.9933 | 0.9678 | 0.5802 |
| problem_073_search_alpha2_seed20260685 | 2 | 20260685 | 9 | 14 | 0.9977 | 0.9744 | 0.5793 |
| problem_074_search_alpha1.5_seed20260686 | 1.5 | 20260686 | 9 | 14 | 2.871 | 2.786 | 0.5763 |
| problem_075_search_alpha2_seed20260686 | 2 | 20260686 | 9 | 15 | 2.96 | 2.854 | 0.5745 |
| problem_076_search_alpha1.5_seed20260689 | 1.5 | 20260689 | 9 | 14 | 3.572 | 3.336 | 0.5873 |
| problem_077_search_alpha2_seed20260689 | 2 | 20260689 | 9 | 14 | 5.064 | 4.238 | 0.5966 |
| problem_078_search_alpha1.5_seed20260690 | 1.5 | 20260690 | 9 | 14 | 2.773 | 2.677 | 0.5966 |
| problem_079_search_alpha2_seed20260690 | 2 | 20260690 | 8 | 14 | 3.602 | 3.112 | 0.6013 |
| problem_080_search_alpha2_seed20260691 | 2 | 20260691 | 9 | 14 | 1.008 | 0.9986 | 0.5868 |
| problem_081_search_alpha1.5_seed20260692 | 1.5 | 20260692 | 9 | 14 | 2.822 | 2.788 | 0.59 |
| problem_082_search_alpha2_seed20260692 | 2 | 20260692 | 9 | 14 | 2.865 | 2.828 | 0.5879 |
| problem_083_search_alpha1.5_seed20260695 | 1.5 | 20260695 | 9 | 14 | 0.9618 | 0.9338 | 0.5923 |
| problem_084_search_alpha2_seed20260695 | 2 | 20260695 | 9 | 14 | 1.928 | 1.886 | 0.5928 |
| problem_085_search_alpha1.5_seed20260696 | 1.5 | 20260696 | 9 | 14 | 1.904 | 1.879 | 0.5833 |
| problem_086_search_alpha2_seed20260696 | 2 | 20260696 | 9 | 14 | 4.622 | 4.386 | 0.5895 |
| problem_087_search_alpha1.5_seed20260697 | 1.5 | 20260697 | 9 | 14 | 1.949 | 1.916 | 0.5885 |
| problem_088_search_alpha2_seed20260697 | 2 | 20260697 | 9 | 14 | 2.849 | 2.741 | 0.5915 |
| problem_089_search_alpha2_seed20260698 | 2 | 20260698 | 9 | 15 | 1.112 | 1.073 | 0.5711 |
| problem_090_search_alpha1.5_seed20260699 | 1.5 | 20260699 | 9 | 14 | 2.848 | 2.74 | 0.5874 |
| problem_091_search_alpha2_seed20260699 | 2 | 20260699 | 9 | 14 | 2.861 | 2.755 | 0.5874 |
| problem_092_search_alpha1.5_seed20260700 | 1.5 | 20260700 | 9 | 14 | 1.905 | 1.793 | 0.5808 |
| problem_093_search_alpha2_seed20260700 | 2 | 20260700 | 9 | 14 | 1.906 | 1.793 | 0.5807 |
| problem_094_search_alpha1.5_seed20260702 | 1.5 | 20260702 | 9 | 14 | 1.845 | 1.825 | 0.5795 |
| problem_095_search_alpha2_seed20260702 | 2 | 20260702 | 9 | 14 | 3.78 | 3.494 | 0.5832 |
| problem_096_search_alpha2_seed20260706 | 2 | 20260706 | 9 | 14 | 1.009 | 1.005 | 0.5814 |
| problem_097_search_alpha2_seed20260707 | 2 | 20260707 | 9 | 14 | 2.852 | 2.753 | 0.5832 |
| problem_098_search_alpha1.5_seed20260708 | 1.5 | 20260708 | 9 | 14 | 1.865 | 1.822 | 0.5837 |
| problem_099_search_alpha2_seed20260708 | 2 | 20260708 | 9 | 14 | 1.873 | 1.828 | 0.5833 |
| problem_100_search_alpha1.5_seed20260711 | 1.5 | 20260711 | 9 | 15 | 0.9758 | 0.9051 | 0.5743 |

Each problem directory contains:

- `problem.json`: clauses, seeds, exact brute-force stats, and selection metrics
- `comparison.json`: full paired DQAS and ME-DQAS run logs
- `summary.csv`: one row per algorithm
- `loss_vs_updates.png`, `loss_vs_shots.png`, `batch_min_mean_max.png`, `final_metrics.png`

The comparison is stochastic; use the saved seeds for reproducibility.
