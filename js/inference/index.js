// dependencies ------------------------------------------------------

var utils = require('./utils')
var inference = require('./inference')

// public api --------------------------------------------------------

var faceDetector = {
	readNifti : utils.readNifti,
	loadModel : utils.loadModel,
    inferer : inference.inferer,
}

// exports -----------------------------------------------------------

module.exports = faceDetector
