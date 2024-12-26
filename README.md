# Scribe

![Scribe Banner](assets/banner.png)

Welcome to the **Scribe** repository! This project provides an end-to-end solution for collecting, training, and simulating human-like mouse movements in the browser. It consists of:

1. A **data collection UI** for recording user mouse movements.
2. A **machine learning model** to train and test the collected data.
3. A **Puppeteer plugin** that utilizes the trained model to produce human-like mouse movements.

## Folder Structure

### `ui/` - Data Collection Site

The `ui` folder hosts the web interface for collecting mouse movement data from users.

### `model/` - Model Training and Testing Scripts

The `model` folder contains the necessary scripts for training a model on the collected mouse movement data and testing the model's output.

### `puppeteer-scribe/` - Puppeteer Plugin

The `puppeteer-scribe` folder contains a Puppeteer plugin that integrates a trained model to control mouse movements in automated browser tasks. This plugin sends start and end coordinates to a server (hosting the trained model), receives the generated path, and moves the cursor smoothly along the predicted trajectory.

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/sameelarif/scribe.git
   cd scribe
   ```
2. **Install Dependencies**
   ```bash
   pip install -r model/requirements.txt
   ```

## Usage

### Data Collection

The data I personally collected can be downloaded from Kaggle: https://www.kaggle.com/datasets/sameelarif/mouse-movement-between-ui-elements.

If you'd like, you can run the data collection UI to collect your own data or add it to the existing dataset. To start the data collection interface:

```bash
cd ui
# Install dependencies
npm i
# Run the Next.js server
npm run start
```

After starting the web server, open it in your browser and use it to record your mouse movement data.

### Model Training

Download the dataset to the `model` directory, and rename the `DATA_FILE` variable from `train.py` to your data file's name. To start the training:

```bash
python train.py
```

The model will output to `model/models`, which you can use to power `puppeteer-scribe`.

### Puppeteer Plugin Integration

The `puppeteer-scribe` folder contains an example of how one would integrate the model into a browser environment. The logic can be forked and edited to support Playwright, selenium, or any other web automation library.

To use the plugin, move your saved model's file into the `model` folder. You don't need to replace previous model versions as the server will automatically pick the most recent file.

Assuming you already have dependencies installed, as previously shown, you can run the server:

```bash
cd puppeteer-scribe
# Run the server
python server/server.py
```

In another terminal window, you can run the test script:

```bash
# Install dependencies
npm i
# Run the test file from `examples`
npm test <test-name>
```

## Contributing

If you'd like to contribute, please create a pull request with a description of your changes. Your contributions will be merged as soon as they are approved.

## License

This work is licensed under a Creative Commons Attribution-NonCommercial 4.0 International License.
To view a copy of this license, visit http://creativecommons.org/licenses/by-nc/4.0/.

If you would like to use this project for commercial reasons, please contact me at `me@sameel.dev`.
