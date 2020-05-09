/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */



const LOCAL_URLS = {
  model_fold1: './models/model_fold1/model.json',
  //model_fold2: 'http://localhost:1235/mri-face-detector/js/models/model_fold2/model.json',
  //model_fold3: 'http://localhost:1235/mri-face-detector/js/models/model_fold3/model.json'  
};

class MnistTransferCNNPredictor {
  /**
   * Initializes the MNIST Transfer CNN demo.
   */
  async init(urls) {
    this.urls = urls;
    this.model_fold1 = await loadHostedPretrainedModel(urls);

    // Print model summary right after model is loaded.
    this.model_fold1.summary();
    tfVis.show.modelSummary(
        {name: 'Model Summary', tab: 'Model Info'}, this.model_fold1);

    this.imageSize = this.model_fold1.layers[0].batchInputShape[1];
    this.numClasses = 2;
    console.log(this.imageSize, this.numClasses);
    return this;
  }


  // Perform prediction on the input image using the loaded model.
  predict(imageText) {
    tf.tidy(() => {
      try {
        const image = util.textToImageArray(imageText, this.imageSize);
        const predictOut = this.model.predict(image);
        const winner = predictOut.argMax(1);

        ui.setPredictResults(predictOut.dataSync(), winner.dataSync()[0] + 5);
      } catch (e) {
        ui.setPredictError(e.message);
      }
    });
  }
}


/**
 * Loads the pretrained model and metadata, and registers the predict
 * and retrain functions with the UI.
 */
async function setupMnistTransferCNN() {

  console.log("hello");
  if (await urlExists(LOCAL_URLS.model_fold1)) {
    console.log('Model available: ' + LOCAL_URLS);
    const button = document.getElementById('load-pretrained-local');
    button.addEventListener('click', async () => {
      const predictor = await new MnistTransferCNNPredictor().init(LOCAL_URLS);
      ui.prepUI(
          x => predictor.predict(x), () => predictor.retrainModel(),
          predictor.testExamples, predictor.imageSize);
    });
    button.style.display = 'inline-block';
  }

  console.log('Standing by. Please load pretrained model first.');
}

setupMnistTransferCNN();
