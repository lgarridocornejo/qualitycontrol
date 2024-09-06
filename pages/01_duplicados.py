import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import qualitycontrol as qc


DEMO = True

st.set_page_config(
    page_title="Duplicados",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# # paleta_discreta= px.colors.carto.Safe #["#1ea47e","#e33f2b",'#fbac5d','#82858d','#4ce1ee','#ffa111']
# # paleta_continua= px.colors.sequential.Jet #['#2effc3','#25c99a','#105743']
# # #paleta_personalizada

st.title("Análisis de Duplicados")
if DEMO:
    st.subheader('DEMO')

file_upload = st.file_uploader("Subir archivo para analisis",type=['csv'])

if file_upload:

    df = pd.read_csv(file_upload)

    if DEMO:
        idle, df = train_test_split(df, test_size=.1, random_state=42)
    #st.write(df.shape[0])


    tab1, tab2 = st.tabs(["Análisis", "Gráficos"])
    GRAFICO = False

    with tab1:
        st.subheader("Tabla de datos")
        c1, c2 = st.columns([80,60])
        with c1:
            st.write(df)
        with c2:
            st.write(df.describe())

        columnas = df.columns
        panalito = ['Ag','Au','Cu']
        punidad = ['%','ppm','ppb','g/t']

    # Seleccionar atributo  *******************************************************************************************************************
        
        st.subheader("Seleccionar atributos")
        c1, c2 = st.columns(2)
        with c1:
            featorgid = st.selectbox('Sampleid original',options=columnas,
                                     index=None,
                                    placeholder="Seleccionar número muestra original")
            featorg = st.selectbox('Muestra original',options=columnas,
                                   index=None,
                                    placeholder="Seleccionar muestra original")
        with c2:
            featdupid = st.selectbox('Sampleid duplicado',options=columnas,
                                     index=None,
                                    placeholder="Seleccionar número muestra duplicado")
            featdup = st.selectbox('Muestra duplicado',options=columnas,
                                   index=None,
                                    placeholder="Seleccionar muestra duplicado")
        

        if featorgid and featorg and featdupid and featdup:
            dup = qc.duplicados(df,featorgid,featorg,featdupid,featdup)
            dup.proceso()

    # Seleccionar analito y unidad  *******************************************************************************************************************
        
            st.subheader("Seleccionar")

            c1, c2 = st.columns(2)
            with c1:
                analito = st.selectbox('Analito',options=panalito,
                                    index=None,
                                        placeholder="Seleccionar analito")
                
            with c2:
                unidad = st.selectbox('Unidad',options=punidad,
                                    index=None,
                                        placeholder="Seleccionar unidad")

            analito_unidad = ''
            if analito:
                analito_unidad = analito

                if unidad:
                    analito_unidad = analito_unidad + ' (' + unidad + ')'  
            
            if not analito:
                analito =''  

            if not unidad:
                unidad =''           
                  

    # # Métricas  *******************************************************************************************************************

            st.subheader("Métricas")

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric(f"Total muestras",f"{df.shape[0]}")   
            
            with c2:
                mse = mean_squared_error(df[featorg], df[featdup])
                st.metric(f"MSE - Error cuadrático medio",f"{mse:.2f}") 
                with st.expander("Ver Explicación"):
                    st.markdown("""Representa el promedio de los cuadrados de las diferencias entre los valores predichos y los valores reales. 
                                Un valor de MSE bajo indica que el modelo utilizado para predecir los valores de las muestras duplicadas 
                                a partir de la original es preciso.
                                """)
                                

            with c3:
                rmse = mean_squared_error(df[featorg], df[featdup], squared=False)
                st.metric(f"RMSE - Raíz del error cuadrático medio",f"{rmse:.2f} " + unidad) 
                with st.expander("Ver Explicación"):
                    st.markdown("""Es simplemente la raíz cuadrada del MSE. 
                                Al estar en las mismas unidades que los datos originales, 
                                proporciona una interpretación más intuitiva del error. 
                                Un RMSE bajo indica un menor error de predicción.
                                """)
                                

            with c4:
                r2 = r2_score(df[featorg], df[featdup])*100
                st.metric(f"R2 - Coeficiente determinación",f"{r2:.1f} %")             
                with st.expander("Ver Explicación"):
                    st.markdown("""Indica la proporción de la variabilidad de los datos que es explicada por el modelo. 
                                Un valor de R² cercano a 100 indica que el modelo explica casi toda la variabilidad de los datos, 
                                mientras que un valor cercano a 0 indica que el modelo no explica prácticamente ninguna variabilidad.
                                """)

                    st.markdown("""- R² < 50 → Coeficiente débil
- 50 ≤ R² ≤ 80 → Coeficiente moderado
- R² > 80 → Coeficiente fuerte

Básicamente significa que, los valores duplicados no se desvían mucho de sus datos originales.""")
            
            GRAFICO = True

# ****************************************************************************************************************************************** 
# ******************************************************************************************************************************************
    with tab2:
        
        if GRAFICO:

    # Tipo de duplicado  *******************************************************************************************************************
            st.subheader("Duplicados")
            tipo_dup = st.radio(
                "¿Qué tipo de duplicado es?",
                ["Gemelo","Grueso","Pulpa"],
                horizontal=True
                )

            if tipo_dup == "Gemelo":
                ER_lin = 30
            elif tipo_dup == "Grueso":
                ER_lin = 20
            else:
                ER_lin = 10

    # Gráficos   


            TOL = st.slider("Escoger la tolerancia",min_value=0, max_value=100, value=ER_lin, step=5)

            # Tamaño grafico
            DIM = 600
            MAX = dup.dfqc['max'].max()
        # ordenadax = np.array([0, MAX+5])
            y_x_color = "#b8b799"
            tol_color = "#cc0605"

            # st.write("MAX:",MAX)

            ordenada =  np.arange(0,MAX+5,0.05)
            abscisa = []
            abscisa1 = []
            for x in ordenada:
                m= 100*x/(100-TOL)
                m1 = (100-TOL)/100*x
                if m < MAX+5:
                    abscisa.append(m)
                abscisa1.append(m1)
            abscisa = np.array(abscisa)

            dup.tol_lineal(TOL)
            cuenta = dup.dfqc['lineal'].value_counts()['Fallo']
            tasa_e = round((cuenta/df.shape[0])*100,2)
            
    # # 1  *******************************************************************************************************************

            c1, c2 = st.columns(2)
            with c1:
                x_etiqueta = 'Original ' + analito_unidad   
                y_etiqueta = 'Duplicado ' + analito_unidad
                st.write("🤖")
                fig = px.scatter(dup.dfqc, 
                                x=dup.dfqc[featorg], 
                                y=dup.dfqc[featdup], 
                                color=dup.dfqc['lineal'],
                                labels={featorg: x_etiqueta, featdup: y_etiqueta, "lineal":"Muestras"},
                                width=DIM,
                                height=DIM,
                                color_discrete_sequence= ['blue','red'], #['blue','red'], {'Fallo':'red','Muestra':'blue'} ,
                                title="Original vs Duplicado"
                                )
                # recta y=x
                fig.add_scatter(x=ordenada, y=ordenada, mode='lines', showlegend=False, line=dict(color=y_x_color))
                fig.add_scatter(x=ordenada, y=abscisa, mode='lines', showlegend=False, line=dict(color=tol_color))
                fig.add_scatter(x=ordenada, y=abscisa1, mode='lines', showlegend=False, line=dict(color=tol_color))
                fig.update_xaxes(showgrid=True)
                fig.update_yaxes(showgrid=True)

                st.plotly_chart(fig)#,use_container_width=False)

            with c2:
                x_etiqueta = 'Min ' + analito_unidad
                y_etiqueta = 'Max ' + analito_unidad 
                st.write("Muestras:",df.shape[0],"Fallos:",cuenta, "Tasa de error:",tasa_e,"%")
                fig = px.scatter(dup.dfqc, 
                                x=dup.dfqc['min'], 
                                y=dup.dfqc['max'], 
                                color=dup.dfqc['lineal'],
                                labels={"min": x_etiqueta, "max": y_etiqueta, "lineal":"Muestras"},
                                width=DIM,
                                height=DIM,                        
                                color_discrete_sequence=['blue','red'],
                                title="Máximo vs Mínimo"
                                )
                # recta y=x
                fig.add_scatter(x=ordenada, y=ordenada, mode='lines', showlegend=False, line=dict(color=y_x_color)) 
                fig.add_scatter(x=ordenada, y=abscisa, mode='lines', showlegend=False, line=dict(color=tol_color))
                fig.update_xaxes(showgrid=True)
                fig.update_yaxes(showgrid=True)

                st.plotly_chart(fig)

    # # 2  *******************************************************************************************************************

            Q75 = dup.dfqc['media'].quantile(0.75)
            Q90 = dup.dfqc['media'].quantile(0.90)
            # st.write(round(Q75,2))
            # st.write(round(Q90,5))
            LPD = st.slider("Escoger Límite Práctico de Detección",min_value=0.0, max_value=Q90, value=Q75, step=0.01)
            #MAXEREL = dup.dfqc['erel'].max()

            tol = TOL/100
            m = (2+tol)/(2-tol)
            m2 = m**2



            if tipo_dup == "Gemelo":
                    izq, der = 10,20
                    if analito == 'Au' or analito == 'Pt' or analito == 'Mo':
                        valor_b = 20 
                    else:
                        valor_b = 10 
            
            elif tipo_dup == "Grueso":
                    izq, der = 5,10
                    if analito == 'Au' or analito == 'Pt' or analito == 'Mo':
                        valor_b = 10 
                    else:
                        valor_b = 5 
            else:
                    izq, der = 3,5
                    if analito == 'Au' or analito == 'Pt' or analito == 'Mo':
                        valor_b = 5 
                    else:
                        valor_b = 3 

            txt_B = "Escoger factor de apertura para : " + analito
            B = st.slider(txt_B,min_value=1, max_value=50, value=valor_b, step=1)
            #B = st.slider(txt_B,min_value=3, max_value=20, value=[izq,der], step=1)
            with st.expander("Ver Explicación"):
                st.markdown(""" """)

            b = B * LPD
            b2 = b**2
            abscisa_hip = (m2 * ordenada**2 + b2)**0.5

            dup.tol_hiperb(m2,b2)

            try:
                cuentah = dup.dfqc['hiperb'].value_counts()['Fallo']
                tasa_eh = round((cuentah/df.shape[0])*100,2)
            except KeyError:
                cuentah = 0
                tasa_eh = 0
            # st.write("m:",round(m,2),"b:",round(b,2))
            # st.write("Muestras:",df.shape[0],"Fallos:",cuentah, "Tasa de error:",tasa_eh,"%")
            # st.write(cuentah)



            c1, c2= st.columns(2)
            with c1:
                x_etiqueta = analito_unidad
                y_etiqueta = 'Error relativo' 
                st.write("🤖")
                fig = px.scatter(dup.dfqc, 
                                x=dup.dfqc['media'], 
                                y=dup.dfqc['erel'], 
                                #color=dup.dfqc['hiperb'],
                                labels={"min": x_etiqueta, "erel": y_etiqueta},
                                width=DIM,
                                height=DIM,     
                                color_discrete_sequence=['blue','red'],
                                title="Límite Práctico de Detección"
                                )
                #fig.add_scatter(x=ordenada, y=ordenada, mode='lines', showlegend=False, line=dict(color="#008f39"))
                #fig.add_scatter(x=LPD, y=1, mode='lines', showlegend=False, line=dict(color="#cc0605"))
                fig.add_vline(x=LPD, line_width=3, line_dash="dash", line_color="#cc0605")
                fig.update_xaxes(showgrid=True)
                fig.update_yaxes(showgrid=True)
                fig.update_xaxes(type="linear",range=[0,Q90])

                st.plotly_chart(fig)

            with c2:
                x_etiqueta = 'Min ' + analito_unidad
                y_etiqueta = 'Max ' + analito_unidad 
                st.write("Muestras:",df.shape[0],"Fallos:",cuentah, "Tasa de error:",tasa_eh,"%")
                fig = px.scatter(dup.dfqc, 
                                x=dup.dfqc['min'], 
                                y=dup.dfqc['max'], 
                                color=dup.dfqc['hiperb'],
                                labels={"min": x_etiqueta, "max": y_etiqueta, "hiperb":"Muestras"},
                                width=DIM,
                                height=DIM,     
                                color_discrete_sequence=['blue','red'],
                                title="Gráfico Máximo vs Mínimo"
                                )
                fig.add_scatter(x=ordenada, y=ordenada, mode='lines', showlegend=False, line=dict(color=y_x_color))
                fig.add_scatter(x=ordenada, y=abscisa_hip, mode='lines', showlegend=False, line=dict(color=tol_color))
                # fig.add_annotation(text='South Korea: Asia <br>China: Asia <br>Canada: North America', 
                #     align='left',
                #     showarrow=False,
                #     xref='paper',
                #     yref='paper',
                #     x=1.1,
                #     y=0.8,
                #     bordercolor='black',
                #     borderwidth=1)
                fig.update_xaxes(showgrid=True)
                fig.update_yaxes(showgrid=True)

                st.plotly_chart(fig)
                with st.expander("Ver Explicación"):
                    st.write("""evaluación de  la precisión por el método hiperbólico (Simón, 2004)""")
                    st.write("m:",round(m,2),"b:",round(b,2))

            # st.write(dup.dfqc)

