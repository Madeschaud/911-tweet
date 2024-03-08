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
	@python -c 'from tweet_911.Model.lstm import model_lstm; model_lstm()'

model_deep_simple_gru:
	@python -c 'from tweet_911.Model.simple_gru import GRU_model; GRU_model()'

model_deep_bidirection_lstm:
	@python -c 'from tweet_911.Model.bidirection_lstm import model_bidirectional_lstm; model_bidirectional_lstm()'

model_deep_cnn_rnn:
	@python -c 'from tweet_911.Model.cnn_rnn import cnn_rnn; cnn_rnn()'

run_cnn_model:
	@python -c 'from tweet_911.Model.cnn import model_cnn; model_cnn()'

save_model:
	@python -c 'from registry import save_model; save_model()'

install_dep:
	pip install -r 'requirements.txt'

save_model:
	@python -c 'from registry import save_model, save_results; save_model(), save_results()'

load_model:
	@python -c 'from registry import load_model; load_model()'

train_model:
	@python -c 'from main import train; train()'

reset_local_files:
	rm -rf ${ML_DIR}
	mkdir -p ~/.lewagon/mlops/data/
	mkdir ~/.lewagon/mlops/data/raw
	mkdir ~/.lewagon/mlops/data/processed
	mkdir ~/.lewagon/mlops/training_outputs
	mkdir ~/.lewagon/mlops/training_outputs/metrics
	mkdir ~/.lewagon/mlops/training_outputs/models
	mkdir ~/.lewagon/mlops/training_outputs/params

#API

run_api:
	uvicorn tweet_911.api.fast:app --reload
