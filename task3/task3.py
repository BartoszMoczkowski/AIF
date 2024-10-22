import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

modes_per_class = 2
classes = 2
samples_per_mode = 100
class DataGenerator():

    df = pd.DataFrame()

    lr = 0.5
    hidden_layer_count = 3


    layer_info = []
    layer_activation_func = [
        "sigmoid",
        "sigmoid",
        "sigmoid",
        "sigmoid",

    ]
    weights = []
    biases = []
    func_dict = {
        'sigmoid' : lambda x : 1/(1+np.exp(-x)),
        'heavyside' : lambda x : np.heaviside(x,1/2),
        'sin' : lambda x : (np.sin(x)+1)/2,
        'tanh' : lambda x : (np.tanh(x)+1)/2,
        'sign' : lambda x : (np.sign(x)+1)/2,
        'ReLu' : lambda x : x * (x>0),
        'lReLu' : lambda x : np.where(x > 0, x , x * 0.01),
        'softmax' : lambda x : np.exp(x) / np.sum(np.exp(x), axis=1).reshape(x.shape[0],1)
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


    def gen_layers(self):
        self.layer_info = [2] + self.hidden_layer_count*[1] + [2]
        self.layer_activation_func = ['']*(self.hidden_layer_count+1)

    def generate_weights(self):
        for i in range(len(self.layer_info)-1):
            weights = np.random.rand(self.layer_info[i],self.layer_info[i+1])
            self.weights.append(weights)
            biases = np.random.rand(1,self.layer_info[i+1])
            self.biases.append(biases)

    def activation_func(self,x,i):
        return self.func_dict[self.layer_activation_func[i]](x)

    def deriv_activation_func(self,x,i):
        return self.derivs[self.layer_activation_func[i]](x)

    def forward_prop(self,inputs):
        pre_activations = []
        activations = []
        activations.append(inputs)
        for i in range(len(self.weights)):
            z = inputs@self.weights[i] + self.biases[i]
            pre_activations.append(z)
            a = self.func_dict[self.layer_activation_func[i]](z)
            activations.append(a)
            inputs = a

        predictions = inputs
        return predictions,pre_activations,activations

    def back_prop(self):
        df = self.df.sample(frac=1).reset_index(drop=True)
        inputs = df[["x","y"]].to_numpy()
        classes = df[["class"]].to_numpy().astype(int)
        n = classes.shape[0]

        n_batches = 10
        batch_size = n//n_batches


        for i_b in range(n_batches):

            l_bound = i_b*batch_size
            u_bound = (i_b+1)*batch_size
            batch_inputs = inputs[l_bound:u_bound,:]
            batch_classes = classes[l_bound:u_bound,:]

            n_batch = batch_classes.shape[0]

            classes_encoded = np.zeros((n_batch,2))
            classes_encoded[np.arange(n_batch),batch_classes.T] = 1

            predictions,pre_activations,activations = self.forward_prop(batch_inputs)
            prediction_delta = (predictions - classes_encoded)

            deltaWs = []
            deltaBs = []

            prev_layer_delta = prediction_delta
            for i in range(len(pre_activations)-1,-1,-1):
                    dW = self.lr*1/n * activations[i].T@prev_layer_delta
                    dB = self.lr*1/n * np.sum(prev_layer_delta,axis=0)
                    deltaWs.append(dW)
                    deltaBs.append(dB)
                    if not i ==0:
                        prev_layer_delta = (prev_layer_delta@self.weights[i].T)*(self.deriv_activation_func(pre_activations[i-1],i-1))

            for i in range(len(deltaWs)):
                self.weights[i] -= deltaWs[::-1][i]
                self.biases[i] -= deltaBs[::-1][i]


    def train(self):
        inputs = self.df[["x","y"]].to_numpy()
        inputs = inputs/inputs.max()
        inputs = inputs - np.mean(inputs)
        classes = self.df[["class"]].to_numpy().astype(int)
        n = classes.shape[0]
        classes_encoded = np.zeros((n,2))
        classes_encoded[np.arange(n),classes.T] = 1
        for e in range(1000):
            self.back_prop()
            if e%100==0:
                predictions,pre_activations,activations = self.forward_prop(inputs)
                print(f"cost at {e}",np.mean(np.abs(classes_encoded - predictions)))
        predictions,pre_activations,activations = self.forward_prop(inputs)
        print("final cost",np.mean(np.abs(classes_encoded - predictions)))
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
    st.button("generate new",on_click=dg.generate)
    dg.hidden_layer_count = int(st.number_input(label = "Hidden layer count",min_value=0,max_value=5,value=3,on_change=dg.gen_layers()))
    dg.gen_layers()
    for i in range(dg.hidden_layer_count):
        dg.layer_info[i+1] = int(st.number_input(label = f"Neuron count for hidden layer : {i+1}",min_value=0,max_value=128,value=3))
    for i in range(dg.hidden_layer_count):
        dg.layer_activation_func[i] = str(st.selectbox(f"Activation function for layer {i+1}",("sigmoid","heavyside","sin","tanh","sign","ReLu","lReLu","softmax"),key=i+1245))
    dg.layer_activation_func[dg.hidden_layer_count] = 'softmax'
dg.generate()
dg.generate_weights()
print(list(map(lambda x : f'{x.shape}',dg.weights)))
print(dg.layer_activation_func)
dg.back_prop()
dg.train()
fig = go.Figure()

x_n = 40
y_n = 40

x = np.linspace(-1.5,1.5,x_n)
y = np.linspace(-1.5,1.5,y_n)

X,Y = np.meshgrid(x,y)

x_f = X.flatten()
y_f = Y.flatten()
heatmap_data = np.vstack([x_f,y_f]).T
p,_,_ = dg.forward_prop(heatmap_data)
decision_map = np.argmax(p,1).reshape(x_n,y_n   )


x1 = dg.df["x"].iloc[0:modes_per_class*samples_per_mode]
y1 = dg.df["y"].iloc[0:modes_per_class*samples_per_mode]
x2 = dg.df["x"].iloc[modes_per_class*samples_per_mode:]
y2 = dg.df["y"].iloc[modes_per_class*samples_per_mode:]
fig = go.Figure(data=go.Heatmap(
                  z=decision_map,dx=3/x_n,dy=3/y_n,x0=-1.5,y0=-1.5,colorscale=[[0, 'rgb(0,0,128)'], [1, 'rgb(128,0,0)']]))
fig.add_trace(go.Scatter(x=x1, y=y1, mode='markers', name='Class 0', marker=dict(color="blue")))
fig.add_trace(go.Scatter(x=x2, y=y2, mode='markers', name='Class 1', marker=dict(color="red")))


st.plotly_chart(fig)
