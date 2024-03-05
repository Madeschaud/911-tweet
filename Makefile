run_preprocess:
	@python -c 'from Preprocessing.preprocessor import preprocessor_all; preprocessor_all()'

run_main:
	@python -c 'from main import main; main()'

run_modify_language:
	@python -c 'from Modify_language.modify_language import modify_language; modify_language()'

install_dep:
	pip install -r requirements.txt
