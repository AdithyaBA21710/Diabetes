# Diabetes
An ML model built to calculate diabetes, based on factors like age, gender, cholesterol, blood pressure, BMI, etc.

This model uses linear regression, with batch gradient descent to predict values. The dataset is part of sklearn's huge repository.

Analysing the dataset by plotting a scatter plot using pandas:
<img width="782" height="681" alt="image" src="https://github.com/user-attachments/assets/2f5911c7-015c-4142-9a53-316cf4085a98" />

Guide to the abbreviations: <br>
s1- Total serum cholesterol <br>
s2- LDL <br>
s3- HDL <br>
s4- Total Cholesterol <br>
s5- Serum triglycerides <br>
s6- Blood sugar level <br>


Error minimisation using gradient descent:
<img width="802" height="680" alt="image" src="https://github.com/user-attachments/assets/d08f41ad-3a52-44b7-b80a-6b69889e3dad" />

When Ridge regression is applied to the problem, the MSE reduces slightly, while the model predicts better than before (without L2 regression)
<img width="1437" height="307" alt="image" src="https://github.com/user-attachments/assets/b4bce2fc-de07-48ef-a806-d6b1f45807fb" />


