# Empanada

## Setup

### Hardware

#### 1. Navigate to the EC2 Dashboard

- **Log in** to your AWS Management Console.
- From the **Services** menu, select **EC2** under the "Compute" category.

---

#### 2. Launch a New Instance

- Click on the **Launch Instances** button on the EC2 dashboard.

---

#### 3. Configure Basic Instance Details

##### 3.1. Name and Tags

- In the **Name and tags** section:
  - Click **Add Tag**.
  - Set **Key** to `Name` and **Value** to `preble-dev`.

##### 3.2. Application and OS Images (Amazon Machine Image)

- Under **Application and OS Images (Amazon Machine Image)**:
  - Click on **Browse more AMIs**.
  - In the search bar, enter the **AMI ID**: `ami-015c62e8068dd8f78`.
  - Select the AMI that matches this ID.

##### 3.3. Instance Type

- In the **Instance type** section:
  - Select **g4dn.xlarge** from the dropdown menu.

##### 3.4. Key Pair (Login)

- Under **Key pair (login)**:
  - Choose an existing key pair or create a new one for SSH access.

---

#### 4. Configure Network Settings

- Expand the **Network settings** section.

##### 4.1. VPC and Subnet

- **VPC**: Leave the default VPC selected.
- **Subnet**: Leave the default subnet selected.

##### 4.2. Auto-assign Public IP

- Ensure **Auto-assign public IPv4 address** is **Enabled**.

##### 4.3. Firewall (Security Groups)

- Under **Firewall (security groups)**:
  - Select **Create security group**.
  - Ensure that the security group contains an **All traffic** rule.

---

#### 5. Configure Storage

- In the **Configure storage** section:
  - For the root volume `/dev/sda1`:
    - **Size (GiB)**: Set to `250`.
    - **Volume Type**: Select **General Purpose SSD (gp3)**.

---

##### 7. Review and Launch

- **Review** all configurations to ensure they match the settings above.
- Click on **Launch Instance** at the bottom of the page.

---

##### 8. Verify Your Instance

- Navigate back to the **EC2 dashboard**.
- Click on **Instances** in the sidebar.
- Confirm that your instance named `preble-dev` is running and has the correct configurations.

---

### Software

#### 1. Install Miniconda

Install Miniconda by running the following commands:

```bash
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash ~/Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc
```

#### 2. Cloning the repository

Clone the current repository by running `git clone https://github.com/inakineitor/preble.git`.

#### 3. Set up the Conda environment

Set up the Conda environment and python packages by running:

1. `cd preble`
1. `source setup_project.sh`

## Old Preble Content

Preble is a load balancer for effecient prefix caching systems.
PrePrint release at https://arxiv.org/abs/2407.00023

## Installation

You can install the package using pip:

# Code Structure

The `multi_node` directory contains the code for running as a separate abstraction layer to SGLang/vLLM in a distributed setting. This code is responsible for coordinating and managing the execution of the distributed system.

Editable Installation

```
pip3 install -e .
pip install -e "python[all]"
pip install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.3/
```

Regular Pip Installation:

```
pip3 install preble
pip install git+https://github.com/wuklab/preble.git#egg=preble[all]
pip install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.3/
```

We release a custom version of sglang that supports chunked prefill

## Programatically starting the server

We can support providing a list of runtime urls

```
from preble.main import start_server

start_server(
    runtime_selection_policy="custom",
    runtime_urls="http://127.0.0.1:30000/generate,http://127.0.0.1:30001/generate",
    host='127.0.0.1',
    port=8000,
    model="mistralai/Mistral-7B-v0.1"
)
```

We can also support dynamically loading the models to seperate cuda devices

```
from preble.main import start_server_and_load_models

start_server_and_load_models(
    model_name="mistralai/Mistral-7B-v0.1",
    devices=[0, 1],
    host="127.0.0.1",
    port=8000
)
```

The server can be run via:

```
python3 multi_node/server/server.py <server/deploy_and_run>
```

- server runs the server given a list of urls
- deploy_and_run generates two endpoints

CLI Configuration

```
    runtime_selection_policy: The policy to select the runtime (e.g., custom, round_robin).
    runtime_urls: Comma-separated list of runtime URLs.
    host: The host address for the server.
    port: The port number for the server.
    model: The model to be used (e.g., mistralai/Mistral-7B-v0.1).
```

## Citation And Acknowledgment

The code is forked of sglang

# pypi build and install instructions

Currently uploaded at:
`python setup.py bdist_wheel`
` twine upload --repository testpypi dist/* --verbose`
`python3 -m pip install --index-url https://test.pypi.org/simple/ preble`

License

This project is licensed under the Apache 2.0 License. See the LICENSE file for details.
