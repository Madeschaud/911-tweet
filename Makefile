run_preprocess:
	@python -c 'from Preprocessing.preprocessor import preprocessor_all; preprocessor_all()'

run_main:
	@python -c 'from main import main; main()'

run_dowload_new_data:
	@python -c 'from Add_data.new_data import download_data; download_data()'


install_dep:
	pip install -r requirements.txt
