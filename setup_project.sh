conda list--name preble >/dev/null 2>&1

if (($? == 0)); then
	echo "Conda environment does not exist"
	echo "Creating it..."
	# Conda environment does not exist
	conda create -n preble python=3.10
	echo "Conda environment created"
	conda activate preble

	echo "Installing PyPI dependencies..."
	pip3 install -e .
	echo "PyPi dependencies installed"

	echo "Installing local version of SGLang..."
	pip install -e "python[all]"
	echo "Local version of SGLang installed"

	echo "Installing FlashInfer..."
	pip install flashinfer -i https://flashinfer.ai/whl/cu124/torch2.4/
	echo "FlashInfer installed"

	echo "Installing rich"
	pip install rich
	echo "Rich installed"

	echo "Installing matplotlib-backend-kitty"
	pip install matplotlib-backend-kitty
	echo "matplotlib-backend-kitty installed"

	conda deactivate
fi

if [[ "$CONDA_DEFAULT_ENV" != "preble" ]]; then
	echo "Activate conda environment is not preble"
	echo "Activating preble environment..."
	conda activate preble
	echo "Preble environment activated"
fi
