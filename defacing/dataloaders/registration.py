import SimpleITK as sitk
import os
import glob
from time import gmtime, strftime



class Coregistration(object):
    """
        for data preprocessing converts volume into (1x1x1) resolution
        along with t1ce or mask registration

    """
    def __init__(self):
        self.registration_method = sitk.ImageRegistrationMethod()

        # Similarity metric settings.
        self.registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
        self.registration_method.SetMetricSamplingStrategy(self.registration_method.RANDOM)
        self.registration_method.SetMetricSamplingPercentage(0.01)

        self.registration_method.SetInterpolator(sitk.sitkLinear)

        # Optimizer settings.
        self.registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, 
                                                            numberOfIterations=100, 
                                                            convergenceMinimumValue=1e-6, 
                                                            convergenceWindowSize=10)
        self.registration_method.SetOptimizerScalesFromPhysicalShift()

        # Setup for the multi-resolution framework.            
        self.registration_method.SetShrinkFactorsPerLevel(shrinkFactors = [4,2,1])
        self.registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2,1,0])
        self.registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    def resize_sitk_3D(self, image_array, outputSize=None, interpolator=sitk.sitkLinear):
        """
        Resample 3D images Image:
        For Labels use nearest neighbour
        For image use
        sitkNearestNeighbor = 1,
        sitkLinear = 2,
        sitkBSpline = 3,
        sitkGaussian = 4,
        sitkLabelGaussian = 5,
        """
        image = image_array
        inputSize = image.GetSize()
        inputSpacing = image.GetSpacing()
        outputSpacing = [1.0, 1.0, 1.0]
        if outputSize:
            outputSpacing[0] = inputSpacing[0] * (inputSize[0] /outputSize[0]);
            outputSpacing[1] = inputSpacing[1] * (inputSize[1] / outputSize[1]);
            outputSpacing[2] = inputSpacing[2] * (inputSize[2] / outputSize[2]);
        else:
            # If No outputSize is specified then resample to 1mm spacing
            outputSize = [0.0, 0.0, 0.0]
            outputSize[0] = int(inputSize[0] * inputSpacing[0] / outputSpacing[0] + .5)
            outputSize[1] = int(inputSize[1] * inputSpacing[1] / outputSpacing[1] + .5)
            outputSize[2] = int(inputSize[2] * inputSpacing[2] / outputSpacing[2] + .5)
        resampler = sitk.ResampleImageFilter()
        resampler.SetSize(outputSize)
        resampler.SetOutputSpacing(outputSpacing)
        resampler.SetOutputOrigin(image.GetOrigin())
        resampler.SetOutputDirection(image.GetDirection())
        resampler.SetInterpolator(interpolator)
        resampler.SetDefaultPixelValue(0)
        image = resampler.Execute(image)
        return image

    def register_patient(self, moving_image, 
                            fixed_image): 
        """
        moving_images : {'key1': path1, 'key2': path2}
        fixed_image :t1c path
        save_path: save path 
        """

        initial_transform = sitk.CenteredTransformInitializer(fixed_image, 
                                                  moving_image, 
                                                  sitk.VersorRigid3DTransform(), 
                                                  sitk.CenteredTransformInitializerFilter.GEOMETRY)

            
        self.registration_method.SetInitialTransform(initial_transform, inPlace=False)
        final_transform = self.registration_method.Execute(sitk.Cast(fixed_image, sitk.sitkFloat32), 
                                          sitk.Cast(moving_image, sitk.sitkFloat32))
            
            
        moving_resampled= sitk.Resample(moving_image, 
                                        fixed_image, 
                                        final_transform, 
                                        sitk.sitkLinear, 0.0, 
                                        moving_image.GetPixelID())
            
        moving_resized = self.resize_sitk_3D(moving_resampled)
        return sitk.GetArrayFromImage(moving_resized).T
