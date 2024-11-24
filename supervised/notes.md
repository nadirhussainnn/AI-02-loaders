References

https://scikit-learn.org/stable/modules/linear_model.html

Oversampling:
Sometimes, datasets maybe unbalanced, for exp: In Loan dataset, the distibution of classes is as

 - Loan_Status: Y = 422
 - Loan_Status: N = 192

If model does not perform well, we must address this issue

Addressing Oversampling/Undersampling issue:
 - We use some techniques to address this issue. For exp
 1. SMOTE (Synthetic Minority Oversampling Technique)
 2. Borderline-SMOTE
 3. Random oversampling
 4. ADASYN

SMOTE
![alt text](image.png)
![alt text](image-1.png)
![alt text](image-2.png)

Random Oversampling
![alt text](image-3.png)
![alt text](image-4.png)

Borderline - SMOTE
![alt text](image-5.png)
![alt text](image-6.png)

ADAYSN: Adaptive synthetic sampling
![alt text](image-7.png)