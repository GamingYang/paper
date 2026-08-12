# Adaptive Attention Final Controls

Protocol: Batch1-8 source training, Batch9 target development/adaptation and checkpoint selection, Batch10 blind-test reporting only.

Seeds: 42, 7, 123
Epochs: 150

## Aggregate

            label          category  n_seeds  mean_rmse  std_rmse  mean_macro_rmse  std_macro_rmse  mean_mae  std_mae  mean_early_rmse  mean_mid_rmse  mean_late_rmse
         Proposed        full_model        3   0.009906  0.000557         0.008764        0.000476  0.006189 0.000516         0.004083       0.013464        0.021748
       static MMD       mmd_control        3   0.010224  0.001211         0.009144        0.000918  0.006400 0.000506         0.004369       0.013678        0.024115
 w/o adaptive MMD       mmd_control        3   0.010830  0.001687         0.009339        0.001556  0.006630 0.001041         0.004635       0.015017        0.018166
w/o physical base structure_control        3   0.016634  0.002211         0.015895        0.001697  0.011158 0.001185         0.008523       0.022211        0.033174

## Per Seed

            label  seed  best_epoch  target_test_rmse  target_test_macro_rmse  target_test_mae  target_test_early_rmse  target_test_mid_rmse  target_test_late_rmse source_status
         Proposed    42          85          0.009278                0.008216         0.005733                0.003545              0.012223               0.026287        reused
         Proposed     7         135          0.010338                0.008995         0.006085                0.003635              0.014774               0.016522        reused
         Proposed   123          93          0.010103                0.009080         0.006749                0.005068              0.013395               0.022435        reused
       static MMD    42          57          0.011249                0.009830         0.006867                0.004616              0.015464               0.023668        reused
       static MMD     7          55          0.010537                0.009500         0.006469                0.004300              0.014685               0.019551        reused
       static MMD   123          73          0.008887                0.008101         0.005863                0.004192              0.010886               0.029127        reused
 w/o adaptive MMD    42          66          0.010731                0.009038         0.006657                0.004925              0.014989               0.015761        reused
 w/o adaptive MMD     7          44          0.012564                0.011023         0.007657                0.005347              0.017964               0.012964        reused
 w/o adaptive MMD   123         105          0.009195                0.007955         0.005575                0.003633              0.012099               0.025772        reused
w/o physical base    42         150          0.018435                0.017077         0.011617                0.007719              0.025332               0.038179        reused
w/o physical base     7         147          0.014166                0.013951         0.009812                0.008079              0.018826               0.024174        reused
w/o physical base   123         138          0.017300                0.016657         0.012045                0.009769              0.022477               0.037168        reused

Interpretation note: MMD controls are auxiliary regularization controls; they should not be written as the dominant contribution.