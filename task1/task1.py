import streamlit as st
import numpy as np
import pandas as pd



modes_per_class = 2
classes = 2
samples_per_mode = 1000
class DataGenerator():

    df = 0 
    def generate(self):
        data = np.zeros((0,2))
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
                
                partial_data = np.array([rand_data_x,rand_data_y]).T
                colors = np.hstack([colors,np.repeat(color_hex,samples_per_mode)])
                data = np.vstack([data,partial_data])
        colors = colors.reshape((classes*modes_per_class*samples_per_mode,1))
        self.df = pd.DataFrame(data,columns=["x","y"])
        self.df.insert(2,"colors",colors)



dg = DataGenerator()
with st.sidebar:
    modes_per_class = int(st.text_input("modes_per_class",value="2"))
    samples_per_mode = int(st.text_input("samples_per_mode",value="100"))
    st.button("generate new",on_click=dg.generate)
dg.generate()
st.scatter_chart(dg.df,x='x',y='y',color="colors",width=600,height=600)

