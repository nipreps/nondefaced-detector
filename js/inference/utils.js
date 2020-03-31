var Jimp = require('jimp');
var niftijs = require('nifti-reader-js');
var tf = require('@tensorflow/tfjs');


var utils = {
  /**
   * loadModel
   *
   * Given the filename, the function asynchronously loads
   * the model and the callback handles the response
   */

  loadModel: async function(filename, callback) {
    model = await tf.loadModel(filename);
    callback(model);
    return model;
  },

  /**
   * readNifti
   *
   * The function takes a file, checks for NIFTI, reads it
   * and returns the necessary contents. The callback handles errors.
   */

    readNifti: function(file, callback) {
    if (niftijs.isCompressed(file)) {
      file = niftijs.decompress(file);
      console.log('Decompressed');
    }

    if (niftijs.isNIFTI(file)) {
      var niftiHeader = niftijs.readHeader(file);
      console.log(niftiHeader)
      var dimensions = niftiHeader.dims.slice(1, 4).reverse();
      console.log('Dimensions : ' + dimensions);

      var image = niftijs.readImage(niftiHeader, file);
      var imagePixels = dimensions.reduce((prod, ele)=>prod*ele);

      if (image.byteLength==imagePixels) {
        var imageData = new Int8Array(image);
      } else if (image.byteLength==imagePixels*2){
        var imageData = new Int16Array(image);
      } else if (image.byteLength==imagePixels*4){
        var imageData = new Float32Array(image);
      } else if (image.byteLength==imagePixels*8) {
        var imageData = new Float64Array(image);
      } else {
        callback('Error in file data format!');
        return;
      }
      
      if(niftiHeader.littleEndian==false) {
        var imageView = new DataView(image);
        var imageData = new Array();
        for(var i=0;i<imagePixels*2;i+=2) {
          imageData[i/2] = imageView.getInt16(i,false)
        }
      }

    } else {
      callback(`Error! Please provide a valid NIFTI file.`);
      return;
    }

    return {
      image: imageData,
      dimensions: dimensions
    };
  },

    /**
     * preprocess
     *
     */
};
