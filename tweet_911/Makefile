reinstall_package:
	@pip uninstall -y tweet_911 || :
	@pip install -e .

run_preprocess:
	@python -c 'from Preprocessing.preprocessor import preprocessor_all; preprocessor_all()'

run_main:
	@python -c 'from main import main; main()'

run_vectorizer:
	@python -c 'from Preprocessing.vectorizer import vectorizer; vectorizer()'

#Model ML
model_ML_baseline:
	@python -c 'from Model.baseline import baseline_naive; baseline_naive()'

model_ML_naive_base:
	@python -c 'from Model.boost_naive_base import boost_naive_base; boost_naive_base()'

#Model Deep
model_deep_lstm:
	@python -c 'from Model.lstm import model_lstm; model_lstm()'

model_deep_simple_gru:
	@python -c 'from Model.simple_gru import GRU_model; GRU_model()'

model_deep_bidirection_lstm:
	@python -c 'from Model.bidirection_lstm import model_bidirectional_lstm; model_bidirectional_lstm()'


save_model:
	@python -c 'from registry import save_model; save_model()'

install_dep:
	pip install -r requirements.txt

save_model:
	@python -c 'from registry import save_model, save_results; save_model(), save_results()'

load_model:
	@python -c 'from registry import load_model; load_model()'


#API

run_api:
	uvicorn tweet_911.api.fast:app --reload
