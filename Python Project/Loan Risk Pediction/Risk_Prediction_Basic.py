import pandas as pd

from sklearn.linear_model import LogisticRegression

#Target : 0 = Sage (Paid) , 1 = Risky (Default)

data = {
    'income':[20000,30000,50000,80000,15000,25000,40000,90000],
    'loan_amount':[5000,10000,15000,20000,25000,30000,45000,10000],
    'default':[0,0,0,0,1,1,1,0]
}

df = pd.DataFrame(data)

#2.Train the classifier
X = df[['income','loan_amount']]
Y = df['default']
model  = LogisticRegression()
model.fit(X,Y)

#3. Predict for a new Applicant
# person makes 30k and wants 25k loan

new_applicant= pd.DataFrame([[30000,25000]],columns= ['income','loan_amount'])
prediction = model.predict(new_applicant)

probablity = model.predict_proba(new_applicant)[0][1]

print(f"Prediction : {'Risky' if prediction[0] == 1 else 'SAFE'}")
print(f"Risky Probablity : {probablity*100:.2f}%")
