# libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

#loading dataset and first review data
brain = pd.read_csv("human-brain-weight.csv")
print(brain.info())
print(brain.head())
print(brain.isnull().sum())

#rename the columns and duplicated the original data
df = brain[:]
df.rename({"Age Range":"Age"}, axis=1, inplace=True)
df.Age = df.Age.astype(str)
age = {"1":"18 years and above", "2":"0 to 18 years"}
df["Age"] = df["Age"].map(age)
print(df.columns)
print(df["Age"].value_counts())


df["Gender"] = df["Gender"].astype(str)
gender = {"1":"Male", "2":"Female"}
df["Gender"] = df["Gender"].map(gender)
print(df["Gender"].value_counts())

print(df.head())
df.rename(columns={"Head Size(cm^3)" : "Head Size",
                   "Brain Weight(grams)":"Brain Weight"}, inplace=True)
print(df.head())
print(df.info())

#Data visualization part 
fig = px.scatter(df, x="Head Size", y="Brain Weight",
                 hover_data="Age",
                 color="Gender",
                 title="Head Size vs. Brain Weigth",
                 size="Head Size",
                 color_discrete_sequence=["green", "purple"],
                 trendline="ols"
                 )
fig.show()

fig1 = px.box(df,
              x="Gender",
              y="Head Size",
              hover_data="Age",
              color="Gender",
              color_discrete_sequence=["goldenrod", "magenta"],
              title="Head Size according to Gender",
              points="all",
              )
fig1.update_traces(quartilemethod="linear")
fig1.show()


gender_by_headsize = df.groupby(["Gender", "Age"]).agg("mean", 
                                                       "Head Size").round(1).sort_values(by="Head Size",ascending=False)
print(gender_by_headsize)

fig2 = px.pie(df,
              names="Gender",
              values="Brain Weight",
              color="Gender",
              opacity=0.50,
              hover_data="Age",             
              color_discrete_sequence=px.colors.sequential.PuBuGn_r,
              title="Pie Chart for Gender acc. to Brain Weight",
              hole=0.3)
fig2.show()

fig3 = px.violin(df,
                 x="Age",
                 y="Head Size",
                 facet_col="Gender",
                 points="all",
                 color="Age",
                 color_discrete_map={"18 years and above":"Purple",
                        "0 to 18 years":"darkblue"}
                 )
fig3.show()


genders = df.groupby(["Gender", "Age"])["Head Size"].max().round(1)
print(genders)

fig4 = px.histogram(df, x="Head Size",
                    color="Gender",
                    nbins=30,
                    opacity=0.8,
                    title="Head Size Histogram Graph",
                    marginal="box",
                    template="plotly_dark"
                    )

fig4.update_layout(
    barmode="overlay",
    bargap=0.1)

fig4.show()


#correlation matrix
correlation = brain.corr()
print(correlation)


palette = sns.color_palette("viridis", as_cmap=True)
sns.heatmap(correlation, annot=True,
            fmt = ".2f",
            cmap=palette)
plt.title("Correlation Matrix")
plt.show()


#ML modeling for regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics


X=brain.drop("Brain Weight(grams)", axis=1)
y = brain["Brain Weight(grams)"]

print(X.shape)
print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

#linear = LinearRegression()
#linear.fit(X_train, y_train)
#print(linear.intercept_)

#coefficient = pd.DataFrame(linear.coef_, X.columns, columns=["Coefficient"])
#print(coefficient)

#y_pred = linear.predict(X_test)
#sns.regplot(x=y_test, y=y_pred)
#plt.show()


models = {"Linear Regression":LinearRegression(),
          "Ridge":Ridge(),
          "Lasso":Lasso(),
          "RandomForestRegressor":RandomForestRegressor(random_state=42),
          "DecisionTreeRegressor":DecisionTreeRegressor(random_state=42)}


for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    #Evulate the model
    mse = metrics.mean_squared_error(y_test, y_pred)
    mae = metrics.mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = metrics.r2_score(y_test, y_pred)


    print(f'{name}: ')
    print(f"Mean Squared Error (MSE):{mse}")
    print(f"Mean Absolute Error (MAE):{mae}")
    print(f"Root Mean Squared Error:(RMSE):{rmse}")
    print(f"R-Squared(R2):{r2}")
    print("------------------------------------")

    print(f"Train Accuracy:", model.score(X_train, y_train))
    print(f"Test Accuracy:",model.score(X_test, y_test))
