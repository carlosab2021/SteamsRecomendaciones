from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import plotly.express as px
from flask import Flask, request, render_template, jsonify
import pandas as pd
import json

app = Flask(__name__, template_folder=".")
app.config['JSON_AS_ASCII'] = False

# Carga la base de datos desde el archivo CSV
data = pd.read_csv('base_de_datos_con_sentimiento.csv')

@app.route('/')
def select_app():
    return render_template('aplicaciones.html')
# Ruta para recibir datos del formulario y ejecutar la aplicación correspondiente
@app.route('/launch_app', methods=['POST'])
def launch_app():
    selected_app = request.form['selected_app']
    input_data = request.form['input_data']
    print("Selected App:", selected_app)
    print("Input Data:", input_data)
    result = {}

    if selected_app == 'recomendacion_juego':
        result = recomendacion_juego(int(input_data))
    elif selected_app == 'recomendacion_user_juego':
        result = recomendacion_user_juego(input_data)
    else:
        result = {"message": "Aplicación no válida."}

    # Obtener recomendaciones y gráfico interactivo solo si la aplicación es 'recomendacion_juego y recomendacion_user_juego '
    
    if selected_app == 'recomendacion_juego':
        recommended_games = get_recommendations(int(input_data))
        result["Recomendaciones de juegos"] = recommended_games
        # También puedes agregar el gráfico interactivo aquí si es necesario

    # Construir un diccionario con el resultado
    response_data = {"result": result}

    # Devolver el diccionario serializado como JSON
    return response_data


# Preprocesamiento de datos para el sistema de recomendación item-item y user-item
data['review'].fillna('', inplace=True)
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(data['review'])
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
# Función para obtener juegos recomendados y mostrar un gráfico interactivo
def get_recommendations(game_id, cosine_sim=cosine_sim):
    game_index = data[data['item_id'] == game_id].index[0]
    sim_scores = list(enumerate(cosine_sim[game_index]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]  # Excluye el juego en sí (índice 0) y toma los 5 más similares
    game_indices = [i[0] for i in sim_scores]
    
    recommended_games = data['title'].iloc[game_indices].tolist()

    # Crear un DataFrame con los juegos recomendados y sus similitudes
    recommendations_df = pd.DataFrame({
        'Game': recommended_games,
        'Similarity': [sim_scores[i][1] for i in range(5)]
    })

    # Crear un gráfico de barras interactivas con Plotly
    fig = px.bar(recommendations_df, x='Similarity', y='Game', orientation='h', title='Juegos Recomendados')
    
    # Guardar el gráfico en un archivo HTML temporal
    fig.write_html('recommendation_plot.html', include_plotlyjs='cdn')

    return recommended_games

# Función para obtener juegos recomendados basados en un juego dado
def get_user_recommendations(user_id, cosine_sim=cosine_sim):
    game_index = data[data['user_id'] == user_id].index[0]
    sim_scores = list(enumerate(cosine_sim[game_index]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]  # Excluye el juego en sí (índice 0) y toma los 5 más similares
    game_indices = [i[0] for i in sim_scores]
    recommendacion_user_juego = data['title'].iloc[game_indices].tolist()
    
    # Crear un DataFrame con los juegos recomendados y sus similitudes
    recommendations_df = pd.DataFrame({
        'Game': recommendacion_user_juego,
        'Similarity': [sim_scores[i][1] for i in range(5)]
    })

    # Crear un gráfico de barras interactivas con Plotly
    fig = px.bar(recommendations_df, x='Similarity', y='Game', orientation='h', title='Juegos Recomendados')
    
    # Guardar el gráfico en un archivo HTML temporal
    fig.write_html('recommendation_plot.html', include_plotlyjs='cdn')

    return recommendacion_user_juego

# Ruta para obtener recomendación de juegos similares
@app.route('/recomendacion_juego/<int:item_id>', methods=['GET'])
def recomendacion_juego(item_id):
    recommended_games = get_recommendations(item_id)
    if not recommended_games:
        return {"message": "No se encontraron Recomendaciones"}, 404
    return {"recommended_games": recommended_games}

@app.route('/recommendation_plot.html')
def open_graph():
    # Aquí debes cargar el archivo HTML o realizar las acciones necesarias
    # para mostrar el gráfico interactivo.
    return render_template('recommendation_plot.html')

# Ruta para obtener recomendación de juegos similares
@app.route('/recomendacion_user_juego/<int:user_id>', methods=['GET'])
def recomendacion_user_juego(user_id):
    recomendacion_user_juego = get_user_recommendations(user_id)
    if not recomendacion_user_juego:
        return {"message": "No se encontraron Recomendaciones"}, 404
    return {"recomendacion_user_juego": recomendacion_user_juego}

    return result
if __name__ == '__main__':
     app.run (debug=True)

