import streamlit as st
st.beta_set_page_config(layout='centered', initial_sidebar_state='expanded', page_icon='data/favicon.png')

st.set_option('deprecation.showfileUploaderEncoding', False)

st.image('data/logo-01.png', use_column_width=True)

selection = st.sidebar.selectbox('App Mode:', ('Instructions', 'Upload', 'View'))
if selection == 'Instructions':
    @st.cache
    def read_md_file(file):
        with open('instructions.md') as f:
            return '\n'.join(f.readlines())


    st.markdown(read_md_file('instructions.md'))

if selection == 'Upload':
    file = st.file_uploader('Upload an Image:')
    if file is not None:
        from run_detector import predict

        predict(file)

if selection == 'View':
    st.warning(
        'Please note that this app makes use of simulated data as a proof of concept. Please do not use this data to make any meaningful decisions regarding COVID-19.')

    import folium
    from folium.plugins import MarkerCluster
    import pandas as pd
    from streamlit_folium import folium_static
    min_days_ago, max_days_ago = st.slider('Days Ago: ', min_value=1, max_value=10, value=[1, 10])

    vis = st.radio('Visualization Mode', ['Clustered Marker Map', '3D Histogram'])

    if vis == 'Clustered Marker Map':
        full_data = pd.read_csv('data/sample_csv.csv')
        MAP_CENTER = (42.36, -71.06)
        used_data = full_data[full_data['DaysElapsed'] <= max_days_ago]
        used_data = used_data[min_days_ago <= used_data['DaysElapsed']]
        # creates map
        map = folium.Map(location=MAP_CENTER, zoom_start=12)

        marker_cluster = MarkerCluster().add_to(map)

        generate_color = lambda n: (round(28.3 * (n - 1)), 255 - round(28.3 * (n - 1)), 0)

        for index, row in used_data.iterrows():
            folium.Marker(
                location=[row[1], row[2]],
                clustered_marker=True,
                popup=row[3],
                icon=folium.Icon(color='black', icon_color=('#%02x%02x%02x' % generate_color(row[3])))
            ).add_to(marker_cluster)

        folium_static(map)

        show_data = st.checkbox('Show Raw Data')
        if show_data:
            st.dataframe(full_data)


    if vis == '3D Histogram':
        import pydeck as pdk
        import numpy as np

        full_data = pd.read_csv('data/bad_csv.csv')

        used_data = full_data[full_data['DaysElapsed'] <= max_days_ago]
        used_data = used_data[min_days_ago <= used_data['DaysElapsed']]

        st.write(pdk.Deck(
            map_style="mapbox://styles/mapbox/light-v9",
            initial_view_state={
                "latitude": np.mean(used_data['Latitude']),
                "longitude": np.mean(used_data['Longitude']),
                "zoom": 14,
                "pitch": 50,
            },
            layers=[
                pdk.Layer(
                    "HexagonLayer",
                    data=used_data,
                    get_position=["Longitude", "Latitude"],
                    radius=5,
                    get_elevation='Score',
                    elevation_scale=0.15,
                    # elevation_range=[0, 100],
                    pickable=True,
                    extruded=True,
                ),
            ],
        ))
        show_data = st.checkbox('Show Raw Data')
        if show_data:
            st.dataframe(full_data)
