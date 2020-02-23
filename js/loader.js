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


/**
 * Test whether a given URL is retrievable.
 */
async function urlExists(url) {
  console.log('Testing url ' + url);
  try {
    const response = await fetch(url, {method: 'HEAD'});
    return response.ok;
  } catch (err) {
    return false;
  }
}

/**
 * Load pretrained model stored at a remote URL.
 *
 * @return An instance of `tf.Model` with model topology and weights loaded.
 */
async function loadHostedPretrainedModel(urls) {
  console.log('Loading pretrained model from ' + urls);
  try {
    const model_fold1 = await tf.loadLayersModel(urls.model_fold1);
    //const model_fold2 = await tf.loadLayersModel(urls.model_fold2);
    //const model_fold3 = await tf.loadLayersModel(urls.model_fold3);

    console.log('Done loading pretrained model.');
    // We can't load a model twice due to
    // https://github.com/tensorflow/tfjs/issues/34
    // Therefore we remove the load buttons to avoid user confusion.
    disableLoadModelButtons();
    return model_fold1; // model_fold2, model_fold3;
  } catch (err) {
    console.error(err);
    console.log('Loading pretrained model failed.');
  }
}

