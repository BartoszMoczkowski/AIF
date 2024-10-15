import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

modes_per_class = 2
classes = 2
samples_per_mode = 100
class DataGenerator():

    df = 0

    lr = 0.1

    weights = np.random.rand(2,1)
    bias = np.random.rand(1,1)
    activation_func_str = 'sigmoid'
    func_dict = {
        'sigmoid' : lambda x : 1/(1+np.exp(-x)),
        'heavyside' : lambda x : np.heaviside(x,1/2),
        'sin' : lambda x : (np.sin(x)+1)/2,
        'tanh' : lambda x : (np.tanh(x)+1)/2,
        'sign' : lambda x : (np.sign(x)+1)/2,
        'ReLu' : lambda x : x * (x>0),
        'lReLu' : lambda x : np.where(x > 0, x , x * 0.01)
    }
    derivs = {
        'sigmoid' : lambda x : 1/(1+np.exp(-x))*(1-1/(1+np.exp(-x))),
        'heavyside' : lambda x : 1,
        'sin' : lambda x : np.cos(x),
        'tanh' : lambda x : 1 - np.tanh(x)**2,
        'sign' : lambda x : 1,
        'ReLu' : lambda x : 1 * (x>0),
        'lReLu' : lambda x : np.where(x > 0, 1 ,  0.01)
    }

    def activation_func(self,x):
        return self.func_dict[self.activation_func_str](x)

    def deriv_activation_func(self,x):
        return self.derivs[self.activation_func_str](x)

    def forward_prop(self):
        inputs = self.df[["x","y"]].to_numpy()

        # a = inputs@self.weights + self.bias
        a = inputs@self.weights
        predictions = self.activation_func(a)
        return predictions

    def back_prop(self):

        inputs = self.df[["x","y"]].to_numpy()
        classes = self.df[["class"]].to_numpy()
        n = classes.shape[0]
        prediction = self.forward_prop()

        prediction_delta = prediction - classes
        weight_change = self.lr*(prediction_delta)*(self.deriv_activation_func(inputs@self.weights))
        #weight_change = self.lr*(prediction_delta)*(self.deriv_activation_func(inputs@self.weights+self.bias))
        self.weights -= 1/n*inputs.T@weight_change

        #[[]]
        #self.bias -= 1/n*np.sum(prediction_delta)

    def train(self):
        classes = self.df[["class"]].to_numpy()
        for e in range(100):
            self.back_prop()
            if e%10==0:
                print(f"cost at {e}",np.mean(np.abs(classes - self.forward_prop())))

        print("final cost",np.mean(np.abs(classes - self.forward_prop())))
    def generate(self):
        data = np.zeros((0,3))
        colors = np.zeros((0))


        for _class in range(classes):
            color = np.random.randint(0,256**3-1)
            color_hex = f"#{color:06x}"

            centre_x = np.random.rand(modes_per_class)*2-1
            centre_y = np.random.rand(modes_per_class)*2-1
            scale_x = np.random.rand(modes_per_class)/5
            scale_y = np.random.rand(modes_per_class)/5
            for mode_i in range(modes_per_class):

                rand_data_x = np.random.normal(loc=centre_x[mode_i],scale=scale_x[mode_i],size=samples_per_mode)
                rand_data_y = np.random.normal(loc=centre_y[mode_i],scale=scale_y[mode_i],size=samples_per_mode)

                partial_data = np.array([rand_data_x,rand_data_y,np.repeat(_class,samples_per_mode)]).T
                colors = np.hstack([colors,np.repeat(color_hex,samples_per_mode)])
                data = np.vstack([data,partial_data])

        colors = colors.reshape((classes*modes_per_class*samples_per_mode,1))
        self.df = pd.DataFrame(data,columns=["x","y","class"])
        self.df.insert(3,"colors",colors)



dg = DataGenerator()
with st.sidebar:
    modes_per_class = int(st.text_input("modes_per_class",value="2"))
    samples_per_mode = int(st.text_input("samples_per_mode",value="100"))
    dg.activation_func_str = st.selectbox("Activation function",("sigmoid","heavyside","sin","tanh","sign","ReLu","lReLu"))
    st.button("generate new",on_click=dg.generate)
dg.generate()
dg.forward_prop()
dg.back_prop()
dg.train()
fig = go.Figure()

x = np.linspace(-1.5,1.5,20)
y = np.linspace(-1.5,1.5,20)

X,Y = np.meshgrid(x,y)

decision_map = dg.activation_func(dg.weights[0,0]*X + dg.weights[1,0]*Y)
#decision_map = dg.activation_func(dg.weights[0,0]*X + dg.weights[1,0]*Y+dg.bias)


x1 = dg.df["x"].iloc[0:modes_per_class*samples_per_mode]
y1 = dg.df["y"].iloc[0:modes_per_class*samples_per_mode]
x2 = dg.df["x"].iloc[modes_per_class*samples_per_mode:]
y2 = dg.df["y"].iloc[modes_per_class*samples_per_mode:]
fig = go.Figure(data=go.Heatmap(
                    z=decision_map,dx=0.15,dy=0.15,x0=-1.5,y0=-1.5,colorscale=[[0, 'rgb(0,0,128)'], [1, 'rgb(128,0,0)']]))
fig.add_trace(go.Scatter(x=x1, y=y1, mode='markers', name='Class 0', marker=dict(color="blue")))
fig.add_trace(go.Scatter(x=x2, y=y2, mode='markers', name='Class 1', marker=dict(color="red")))


st.plotly_chart(fig)
