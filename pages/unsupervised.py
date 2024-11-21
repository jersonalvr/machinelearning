# pages/unsupervised.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import google.generativeai as genai
import umap

def show_unsupervised():
    # Verificar datos preparados
    if 'prepared_data' not in st.session_state or st.session_state.prepared_data is None:
        st.warning("Por favor, carga y prepara tus datos primero en la página de Preparación.")
        return

    # Obtener datos
    data = st.session_state.prepared_data

    # Seleccionar columnas numéricas
    numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()

    if not numeric_cols:
        st.error("No hay variables numéricas para realizar análisis no supervisado.")
        return
    
    # Selección de características
    feature_cols = st.multiselect(
        "Seleccionar Variables para Análisis",
        numeric_cols,
        default=numeric_cols[:min(5, len(numeric_cols))]
    )

    if not feature_cols:
        st.warning("Selecciona al menos una variable.")
        return
    
    # Selección de método con selección múltiple
    metodos = st.multiselect(
        "Seleccionar Métodos",
        [
            "K-Means",
            "DBSCAN",
            "Clustering Jerárquico",
            "Análisis de Componentes Principales (PCA)",
            "t-SNE",
            "UMAP"
        ],
        key='metodos'
    )

    # Variables para almacenar métricas globales
    global_silhouette = None
    global_calinski = None
    global_davies = None
    method_details = {}

    # Calcular número de columnas dinámicamente
    if metodos:
        # Crear columnas con métodos seleccionados
        cols = st.columns(len(metodos))
        
        # Iterar sobre métodos seleccionados
        for i, metodo in enumerate(metodos):
            with cols[i]:
                # Contenedor para cada método
                with st.container():
                    st.subheader(f"Resultados de {metodo}")

                    # Configuraciones específicas por método
                    n_clusters = None
                    init_method = None
                    eps = None
                    min_samples = None
                    n_components = None

                    # Parámetros específicos por método
                    if metodo in ["K-Means", "Clustering Jerárquico"]:
                        n_clusters = st.slider(f"Número de Clusters ({metodo})", 2, 10, 3, key=f'n_clusters_{i}')

                    if metodo == "K-Means":
                        init_method = st.selectbox(
                            "Método de Inicialización", 
                            ["k-means++", "random"], 
                            key=f'init_method_{i}'
                        )

                    if metodo == "DBSCAN":
                        eps = st.slider(f"Epsilon ({metodo})", 0.1, 2.0, 0.5, key=f'eps_{i}')
                        min_samples = st.slider(f"Mínimo de Muestras ({metodo})", 2, 20, 5, key=f'min_samples_{i}')

                    if metodo in ["t-SNE", "UMAP"]:
                        n_components = st.slider(f"Número de Componentes ({metodo})", 2, 3, 2, key=f'n_components_{i}')

                    # Preparar datos
                    X = data[feature_cols]
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)

                    # Construir texto de parámetros
                    params_text = "Los parámetros utilizados en el método son:\n"

                    if metodo == "K-Means":
                        params_text += f"- Número de Clusters: {n_clusters}\n"
                        params_text += f"- Método de Inicialización: {init_method}\n"
                    elif metodo == "DBSCAN":
                        params_text += f"- Epsilon (eps): {eps}\n"
                        params_text += f"- Mínimo de Muestras por Cluster: {min_samples}\n"
                    elif metodo == "Clustering Jerárquico":
                        params_text += f"- Número de Clusters: {n_clusters}\n"
                    elif metodo == "Análisis de Componentes Principales (PCA)":
                        params_text += "No se especificaron parámetros adicionales.\n"
                    elif metodo == "t-SNE":
                        params_text += f"- Número de Componentes: {n_components}\n"
                    elif metodo == "UMAP":
                        params_text += f"- Número de Componentes: {n_components}\n"
                    else:
                        params_text += "No se especificaron parámetros adicionales.\n"

                    # Procesamiento según método
                    try:
                        if metodo == "K-Means":
                            modelo = KMeans(
                                n_clusters=n_clusters,
                                random_state=42,
                                n_init=10,
                                init=init_method
                            )
                            clusters = modelo.fit_predict(X_scaled)
                            
                            # Métricas de clustering
                            silhouette = silhouette_score(X_scaled, clusters)
                            calinski = calinski_harabasz_score(X_scaled, clusters)
                            davies = davies_bouldin_score(X_scaled, clusters)

                            # Almacenar métricas globales
                            global_silhouette = silhouette
                            global_calinski = calinski
                            global_davies = davies

                            st.subheader("Métricas de Clustering")
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Silhouette Score", f"{silhouette:.4f}")
                            with col2:
                                st.metric("Calinski-Harabasz", f"{calinski:.2f}")
                            with col3:
                                st.metric("Davies-Bouldin", f"{davies:.4f}")

                            # Visualización de clusters
                            pca = PCA(n_components=2)
                            X_pca = pca.fit_transform(X_scaled)
                            fig = px.scatter(
                                x=X_pca[:, 0],
                                y=X_pca[:, 1],
                                color=clusters.astype(str),
                                title=f"Clusters {metodo}",
                                labels={'x': 'PCA 1', 'y': 'PCA 2'}
                            )
                            st.plotly_chart(fig)

                            with st.expander("🧠 Interpretación de K-Means"):
                                st.markdown("""
                                    ### Objetivo: Dividir los datos en K grupos distintos

                                    **Características:**
                                    - Cada punto pertenece al cluster con centro más cercano
                                    - Minimiza la varianza dentro de cada cluster
                                    - Los centroides representan el centro de cada grupo

                                    **Métricas Clave:**
                                    - **Silhouette Score:** Mide qué tan similar es un punto a su propio cluster comparado con otros clusters
                                        - Rango: -1 a 1 (mayor es mejor)
                                        - > 0.7: Estructura fuerte
                                        - 0.5-0.7: Estructura razonable
                                        - < 0.5: Estructura débil
                                    - **Calinski-Harabasz:** Evalúa la separación entre clusters
                                        - Mayor valor indica mejor definición de clusters
                                    - **Davies-Bouldin:** Mide la similitud promedio entre cada cluster
                                        - Menor valor indica mejor separación entre clusters

                                    **Interpretación Práctica:**
                                    - **Número de Clusters (K):** Define cuántos grupos queremos encontrar
                                    - **Inicialización:** Afecta la posición inicial de los centroides
                                    - Útil para segmentación de clientes, agrupación de productos, etc.
                                    """)

                        elif metodo == "DBSCAN":
                            modelo = DBSCAN(eps=eps, min_samples=min_samples)
                            clusters = modelo.fit_predict(X_scaled)
                            
                            # Count unique clusters (excluding noise points)
                            unique_clusters = np.setdiff1d(np.unique(clusters), [-1])
                            
                            if len(unique_clusters) > 1:
                                # Metrics for non-noise clusters
                                non_noise_mask = clusters != -1
                                X_scaled_clustered = X_scaled[non_noise_mask]
                                clusters_clustered = clusters[non_noise_mask]
                                
                                # Calculate metrics
                                silhouette = silhouette_score(X_scaled_clustered, clusters_clustered)
                                calinski = calinski_harabasz_score(X_scaled_clustered, clusters_clustered)
                                davies = davies_bouldin_score(X_scaled_clustered, clusters_clustered)

                                st.subheader("Métricas de Clustering")
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Silhouette Score", f"{silhouette:.4f}")
                                with col2:
                                    st.metric("Calinski-Harabasz", f"{calinski:.2f}")
                                with col3:
                                    st.metric("Davies-Bouldin", f"{davies:.4f}")

                                # Visualization using PCA
                                pca = PCA(n_components=2)
                                X_pca = pca.fit_transform(X_scaled)
                                
                                # Create color mapping including noise points
                                color_map = clusters.copy().astype(str)
                                color_map[clusters == -1] = 'Noise'
                                
                                fig = px.scatter(
                                    x=X_pca[:, 0],
                                    y=X_pca[:, 1],
                                    color=color_map,
                                    title=f"Clusters {metodo}",
                                    labels={'x': 'PCA 1', 'y': 'PCA 2'}
                                )
                                st.plotly_chart(fig)
                                
                                # Additional information about clustering
                                st.write(f"Total Clusters Found: {len(unique_clusters)}")
                                st.write(f"Noise Points: {np.sum(clusters == -1)}")
                            else:
                                st.warning(f"DBSCAN no pudo encontrar múltiples clusters con los parámetros actuales. Intente ajustar eps ({eps}) o min_samples ({min_samples}).")
                            with st.expander("🧠 Interpretación de DBSCAN"):
                                st.markdown("""
                                    ### Objetivo: Encontrar clusters basados en densidad

                                    **Características:**
                                    - Puede detectar clusters de formas arbitrarias
                                    - Identifica puntos de ruido (outliers)
                                    - No requiere especificar número de clusters

                                    **Parámetros Clave:**
                                    - **Epsilon (eps):** Radio máximo entre puntos del mismo cluster
                                        - Si es muy pequeño: Muchos clusters pequeños/ruido
                                        - Si es muy grande: Pocos clusters grandes
                                    - **Min Samples:** Número mínimo de puntos para formar un cluster
                                        - Afecta la sensibilidad a ruido
                                        - Valores típicos: 2 * n_features a 2 * n_features + 1

                                    **Interpretación Práctica:**
                                    - Ideal para datos con ruido
                                    - Bueno para detectar clusters de formas irregulares
                                    - Los puntos marcados como ruido (-1) son potenciales outliers
                                    """)
                        elif metodo == "Clustering Jerárquico":
                            modelo = AgglomerativeClustering(n_clusters=n_clusters)
                            clusters = modelo.fit_predict(X_scaled)
                            
                            # Métricas de clustering
                            silhouette = silhouette_score(X_scaled, clusters)
                            calinski = calinski_harabasz_score(X_scaled, clusters)
                            davies = davies_bouldin_score(X_scaled, clusters)

                            # Almacenar métricas globales
                            global_silhouette = silhouette
                            global_calinski = calinski
                            global_davies = davies

                            st.subheader("Métricas de Clustering")
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Silhouette Score", f"{silhouette:.4f}")
                            with col2:
                                st.metric("Calinski-Harabasz", f"{calinski:.2f}")
                            with col3:
                                st.metric("Davies-Bouldin", f"{davies:.4f}")

                            # Visualización de clusters
                            pca = PCA(n_components=2)
                            X_pca = pca.fit_transform(X_scaled)
                            fig = px.scatter(
                                x=X_pca[:, 0],
                                y=X_pca[:, 1],
                                color=clusters.astype(str),
                                title=f"Clusters {metodo}",
                                labels={'x': 'PCA 1', 'y': 'PCA 2'}
                            )
                            st.plotly_chart(fig)
                            with st.expander("🧠 Interpretación de Clustering Jerárquico"):
                                st.markdown("""
                                    ### Objetivo: Crear una jerarquía de clusters

                                    **Características:**
                                    - Construye una jerarquía de clusters (dendrograma)
                                    - Puede ser aglomerativo (bottom-up) o divisivo (top-down)
                                    - No requiere especificar número de clusters inicialmente

                                    **Métricas Clave:**
                                    - **Silhouette Score:** Evalúa la calidad de los clusters
                                    - **Calinski-Harabasz:** Mide la separación entre clusters
                                    - **Davies-Bouldin:** Evalúa la similaridad entre clusters

                                    **Interpretación Práctica:**
                                    - Útil cuando se busca una estructura jerárquica
                                    - Permite visualizar relaciones entre grupos
                                    - Ayuda a determinar el número óptimo de clusters
                                    - Ideal para taxonomías y estructuras organizativas
                                    """)
                        elif metodo == "Análisis de Componentes Principales (PCA)":
                            pca = PCA()
                            X_pca = pca.fit_transform(X_scaled)

                            # Varianza explicada
                            varianza_explicada = pca.explained_variance_ratio_

                            # Tabla de varianza explicada
                            varianza_df = pd.DataFrame({
                                'Componente': range(1, len(varianza_explicada) + 1),
                                'Varianza Explicada (%)': varianza_explicada * 100,
                                'Varianza Acumulada (%)': np.cumsum(varianza_explicada) * 100
                            })
                            st.dataframe(varianza_df)

                            # Gráfico de varianza explicada
                            fig_varianza = px.line(
                                varianza_df,
                                x='Componente',
                                y='Varianza Acumulada (%)',
                                title='Varianza Explicada Acumulada'
                            )
                            st.plotly_chart(fig_varianza)
                            with st.expander("🧠 Interpretación de Análisis de Componentes Principales (PCA)"):
                               st.markdown("""
                                    ### Objetivo: Reducir dimensionalidad preservando varianza

                                    **Características:**
                                    - Transforma variables correlacionadas en componentes no correlacionados
                                    - Mantiene la mayor cantidad de información posible
                                    - Ordena componentes por importancia

                                    **Interpretación de Resultados:**
                                    - **Varianza Explicada:** Porcentaje de información retenida
                                        - Suma acumulada ayuda a decidir número de componentes
                                        - Se busca típicamente 80-90% de varianza acumulada
                                    - **Componentes Principales:**
                                        - Primer componente: Dirección de máxima varianza
                                        - Componentes subsecuentes: Ortogonales entre sí

                                    **Aplicaciones Prácticas:**
                                    - Reducción de dimensionalidad
                                    - Visualización de datos multidimensionales
                                    - Eliminación de multicolinealidad
                                    - Compresión de datos
                                    """) 
                        elif metodo == "t-SNE":
                            tsne = TSNE(n_components=n_components, random_state=42)
                            X_tsne = tsne.fit_transform(X_scaled)

                            if n_components == 2:
                                fig_tsne = px.scatter(
                                    x=X_tsne[:, 0],
                                    y=X_tsne[:, 1],
                                    title='Visualización t-SNE',
                                    labels={'x': 't-SNE 1', 'y': 't-SNE 2'}
                                )
                                st.plotly_chart(fig_tsne)
                            else:
                                fig_tsne = go.Figure(data=[
                                    go.Scatter3d(
                                        x=X_tsne[:, 0],
                                        y=X_tsne[:, 1],
                                        z=X_tsne[:, 2],
                                        mode='markers',
                                        marker=dict(
                                            size=5,
                                            color=X_tsne[:, 0],
                                            colorscale='Viridis',
                                            opacity=0.8
                                        )
                                    )
                                ])
                                fig_tsne.update_layout(
                                    title='Visualización t-SNE 3D',
                                    scene=dict(
                                        xaxis_title='t-SNE 1',
                                        yaxis_title='t-SNE 2',
                                        zaxis_title='t-SNE 3'
                                    )
                                )
                                st.plotly_chart(fig_tsne)
                            with st.expander("🧠 Interpretación de t-SNE"):
                                st.markdown("""
                                    ### Objetivo: Visualización de datos de alta dimensión

                                    **Características:**
                                    - Preserva estructura local de los datos
                                    - No lineal: Captura relaciones complejas
                                    - Enfocado en visualización

                                    **Parámetros Importantes:**
                                    - **Perplexidad:** Balance entre estructura local y global
                                        - Rango típico: 5-50
                                        - Afecta la distribución de puntos
                                    - **Número de iteraciones:** Afecta la calidad del resultado
                                    
                                    **Interpretación Práctica:**
                                    - Distancias absolutas no son significativas
                                    - Clusters visibles sugieren grupos naturales
                                    - Útil para exploración visual de datos
                                    - Complementa otros métodos de clustering
                                    """)
                        elif metodo == "UMAP":
                            try:
                                reducer = umap.UMAP(n_components=n_components, random_state=42)
                                X_umap = reducer.fit_transform(X_scaled)

                                if n_components == 2:
                                    fig_umap = px.scatter(
                                        x=X_umap[:, 0],
                                        y=X_umap[:, 1],
                                        title='Visualización UMAP',
                                        labels={'x': 'UMAP 1', 'y': 'UMAP 2'}
                                    )
                                    st.plotly_chart(fig_umap)
                                else:
                                    fig_umap = go.Figure(data=[
                                        go.Scatter3d(
                                            x=X_umap[:, 0],
                                            y=X_umap[:, 1],
                                            z=X_umap[:, 2],
                                            mode='markers',
                                            marker=dict(
                                                size=5,
                                                color=X_umap[:, 0],
                                                colorscale='Plasma',
                                                opacity=0.8
                                            )
                                        )
                                    ])
                                    fig_umap.update_layout(
                                        title='Visualización UMAP 3D',
                                        scene=dict(
                                            xaxis_title='UMAP 1',
                                            yaxis_title='UMAP 2',
                                            zaxis_title='UMAP 3'
                                        )
                                    )
                                    st.plotly_chart(fig_umap)

                            except ImportError:
                                st.error("El paquete UMAP no está instalado. Instálalo con: pip install umap-learn")
                            with st.expander("🧠 Interpretación de UMAP"):
                                st.markdown("""
                                    ### Objetivo: Reducción de dimensionalidad y visualización

                                    **Características:**
                                    - Preserva estructura global y local
                                    - Más rápido que t-SNE
                                    - Mejor preservación de estructura global

                                    **Parámetros Clave:**
                                    - **n_neighbors:** Balance entre estructura local/global
                                    - **min_dist:** Controla la compactación de puntos
                                    - **n_components:** Dimensiones de salida

                                    **Interpretación Práctica:**
                                    - Más escalable que t-SNE
                                    - Mejor para grandes conjuntos de datos
                                    - Preserva más información topológica
                                    - Útil para visualización y clustering
                                    """)
                    except Exception as e:
                        st.error(f"Error en {metodo}: {e}")

        # Sección de Explicación de Parámetros
        st.write("---")
        st.write("### Explicación de Parámetros")

        # Verificar la clave API de Gemini
        has_api_key = 'gemini_api_key' in st.session_state and st.session_state.gemini_api_key

        if not has_api_key:
            st.warning("Configure su API key de Gemini en la sección superior izquierda para usar la explicación automática de los parámetros.")

        # Inicializar las explicaciones en el estado de sesión si no existen
        if 'model_explanations' not in st.session_state:
            st.session_state.model_explanations = {}

        # Crear un botón para generar la explicación
        explain_button = st.button(
            "Explicar Parámetros",
            disabled=not has_api_key,
            key=f"explain_params"
        )

        # Mostrar las explicaciones existentes
        if metodos:
            for method in metodos:
                if method in st.session_state.model_explanations:
                    st.markdown(f"### Explicación de {method}")
                    st.markdown(st.session_state.model_explanations[method])

        if explain_button and has_api_key:
            try:
                with st.spinner("Generando explicación..."):
                    # Configurar Gemini
                    genai.configure(api_key=st.session_state.gemini_api_key)
                    model = genai.GenerativeModel('gemini-1.5-flash')

                    # Generar explicaciones para cada método
                    for method in metodos:
                        # Encontrar los parámetros para este método
                        method_params = [p for p in params_text.split('\n') if method in p or method.lower() in p.lower()]
                        method_params_text = '\n'.join(method_params) if method_params else "No se especificaron parámetros adicionales."

                        # Preparar el prompt
                        prompt = f"""Por favor, proporciona una explicación detallada de los parámetros utilizados en el método de {method} para análisis no supervisado. Incluye ejemplos y explica cómo los diferentes valores de parámetros afectan el resultado.

                        {method_params_text}
                        """

                        # Generar la explicación
                        response = model.generate_content(prompt)

                        # Guardar la explicación en el estado de sesión
                        st.session_state.model_explanations[method] = response.text

                        # Mostrar la explicación
                        st.markdown(f"### Explicación de {method}")
                        st.markdown(response.text)

            except Exception as e:
                st.error(f"Error al generar la explicación: {str(e)}")

        # Botón para guardar resultados
        if st.button("Guardar Resultados"):
            try:
                # Crear DataFrame con resultados
                resultados_df = pd.DataFrame({
                    'Método': metodos,
                    'Variables': [', '.join(feature_cols)] * len(metodos),
                    'Silhouette Score': [global_silhouette] if global_silhouette is not None else [None],
                    'Calinski-Harabasz Score': [global_calinski] if global_calinski is not None else [None],
                    'Davies-Bouldin Score': [global_davies] if global_davies is not None else [None]
                })

                # Guardar CSV
                resultados_df.to_csv('unsupervised_results.csv', index=False)
                st.success("Resultados guardados en 'unsupervised_results.csv'")

            except Exception as e:
                st.error(f"Error al guardar resultados: {e}")