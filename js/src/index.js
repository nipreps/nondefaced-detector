var tf = require('@tensorflow/tfjs');

var path = require('path');
var Jimp = require('jimp');

var inference  = require('../inference');

let model;

const messageElement = document.getElementById('message');
const statusElement = document.getElementById('status');
const imageElement = document.getElementById('images');
const inputElement = document.getElementById('file_input');


function clearElement(element) {
  while (element.firstChild) {
    element.removeChild(element.firstChild);
  }
}

function showResults(result) {
  console.log('Prediction : ' + result);

  var status = `Prediction : ${result[0].toFixed(2)} `;

  statusElement.innerText = status;
  statusElement.innerText += '\n';

  if (result[0] < 0.5) {
    statusElement.innerText += ' It has NOT been defaced.';
  } else {
    statusElement.innerText += ' It has been defaced.';
  }
}


