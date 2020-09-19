import streamlit as st

st.set_option('deprecation.showfileUploaderEncoding', False)

'# Covid Danger Meter'

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
    pass