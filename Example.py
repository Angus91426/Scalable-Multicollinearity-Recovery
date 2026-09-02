import numpy as np
import Scalable_Multicollinearity_Recovery as SMR

# Synthetic example: 2000 x 1000 design with planted relationships
indices, coef, X, feature_col = SMR.Simulation_Data(
    n=2000, p=1000, MR=[0, 5, 3, 1, 1, 0, 0], noise_scale=0.01, rand_seed=911122
)

detector = SMR.Multicollinear(
    reduction=True, reduction_method='eigvec',
    Inequality_Inspection=True, Irreducibility_Inspection=True, fastpath=True
)
relationships = detector.SMR_Main(X=X, col_names=feature_col)
acc, fpr = detector.Multicollinear_score(relationships, indices)
print(f'Accuracy: {acc:.1f}%, False-positive rate: {fpr:.1f}%')
