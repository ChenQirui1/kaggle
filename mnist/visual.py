import plotly.express as px
from sklearn.datasets import load_breast_cancer



X,y = load_breast_cancer(return_X_y=True)
X = X[:, :3]

fig = px.scatter_3d(X, x=0, y=1, z=2, title='3D Scatter Plot of Dataset', color=y)

fig.show()