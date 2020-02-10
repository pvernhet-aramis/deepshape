#include "itkImage.h"
#include "itkImageFileReader.h"

int
main(int argc, char * argv[])
{
  if (argc != 2)
  {
    std::cerr << "Usage: " << std::endl;
    std::cerr << argv[0];
    std::cerr << " <InputFileName>";
    std::cerr << std::endl;
    return EXIT_FAILURE;
  }

  constexpr unsigned int Dimension = 2;

  using PixelType = unsigned char;
  using ImageType = itk::Image<PixelType, Dimension>;

  using ReaderType = itk::ImageFileReader<ImageType>;
  ReaderType::Pointer reader = ReaderType::New();
  reader->SetFileName(argv[1]);
  reader->Update();

  ImageType::Pointer image = reader->GetOutput();

  return EXIT_SUCCESS;
}




//
// /*=========================================================================
//  *
//  *  Copyright Insight Software Consortium
//  *
//  *  Licensed under the Apache License, Version 2.0 (the "License");
//  *  you may not use this file except in compliance with the License.
//  *  You may obtain a copy of the License at
//  *
//  *         http://www.apache.org/licenses/LICENSE-2.0.txt
//  *
//  *  Unless required by applicable law or agreed to in writing, software
//  *  distributed under the License is distributed on an "AS IS" BASIS,
//  *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  *  See the License for the specific language governing permissions and
//  *  limitations under the License.
//  *
//  *=========================================================================*/
//
// #include "itkImageFileReader.h"
// #include "itkImageFileWriter.h"
//
// #include "itkRescaleIntensityImageFilter.h"
// #include "itkHistogramMatchingImageFilter.h"
//
// //  Software Guide : BeginLatex
// //
// // The finite element (FEM) library within the Insight Toolkit can be
// // used to solve deformable image registration problems.  The first step in
// // implementing a FEM-based registration is to include the appropriate
// // header files.
// //
// //  \index{Registration!Finite Element-Based}
//
//
// #include "itkFEMRegistrationFilter.h"
//
//
// //  Next, we use \code{using} type alias to instantiate all necessary classes.  We
// //  define the image and element types we plan to use to solve a
// //  two-dimensional registration problem.  We define multiple element
// //  types so that they can be used without recompiling the code.
//
//
// //  Note that in order to solve a three-dimensional registration
// //  problem, we would simply define 3D image and element types in lieu
// //  of those above.  The following declarations could be used for a 3D
// //  problem:
//
//
// //  Once all the necessary components have been instantiated, we can
// //  instantiate the \doxygen{FEMRegistrationFilter}, which depends on the
// //  image input and output types.
//
//
// // using itk::Image<unsigned char, 2> = itk::Image<unsigned char, 2>;
// // using itk::Image<float, 2> = itk::Image<float, 2>;
// // using itk::ImageFileReader<itk::Image<unsigned char, 2>> = itk::ImageFileReader<itk::Image<unsigned char, 2>>;
// // using itk::RescaleIntensityImageFilter<itk::Image<unsigned char, 2>, itk::Image<float, 2>> = itk::RescaleIntensityImageFilter<itk::Image<unsigned char, 2>, itk::Image<float, 2>>;
// // using itk::HistogramMatchingImageFilter<itk::Image<float, 2>, itk::Image<float, 2>> = itk::HistogramMatchingImageFilter<itk::Image<float, 2>, itk::Image<float, 2>>;
//
// int
// main(int argc, char * argv[])
// {
//   const char *fixedImageName, *movingImageName;
//   if (argc < 2)
//   {
//     std::cout << "Image file names missing" << std::endl;
//     std::cout << "Usage: " << argv[0] << " fixedImageFile movingImageFile" << std::endl;
//     return EXIT_FAILURE;
//   }
//   else
//   {
//     fixedImageName = argv[1];
//     movingImageName = argv[2];
//   }
//
//
//   // Read the image files
//   itk::ImageFileReader<itk::Image<unsigned char, 2>>::Pointer movingfilter = itk::ImageFileReader<itk::Image<unsigned char, 2>>::New();
//   movingfilter->SetFileName(movingImageName);
//   itk::ImageFileReader<itk::Image<unsigned char, 2>>::Pointer fixedfilter = itk::ImageFileReader<itk::Image<unsigned char, 2>>::New();
//   fixedfilter->SetFileName(fixedImageName);
//   std::cout << " reading moving " << movingImageName << std::endl;
//   std::cout << " reading fixed " << fixedImageName << std::endl;
//
//
//   try
//   {
//     movingfilter->Update();
//   }
//   catch (itk::ExceptionObject & e)
//   {
//     std::cerr << "Exception caught during reference file reading " << std::endl;
//     std::cerr << e << std::endl;
//     return EXIT_FAILURE;
//   }
//   try
//   {
//     fixedfilter->Update();
//   }
//   catch (itk::ExceptionObject & e)
//   {
//     std::cerr << "Exception caught during target file reading " << std::endl;
//     std::cerr << e << std::endl;
//     return EXIT_FAILURE;
//   }
//
//
//   // Rescale the image intensities so that they fall between 0 and 255
//   itk::RescaleIntensityImageFilter<itk::Image<unsigned char, 2>, itk::Image<float, 2>>::Pointer movingrescalefilter = itk::RescaleIntensityImageFilter<itk::Image<unsigned char, 2>, itk::Image<float, 2>>::New();
//   itk::RescaleIntensityImageFilter<itk::Image<unsigned char, 2>, itk::Image<float, 2>>::Pointer fixedrescalefilter = itk::RescaleIntensityImageFilter<itk::Image<unsigned char, 2>, itk::Image<float, 2>>::New();
//
//   movingrescalefilter->SetInput(movingfilter->GetOutput());
//   fixedrescalefilter->SetInput(fixedfilter->GetOutput());
//
//   constexpr double desiredMinimum = 0.0;
//   constexpr double desiredMaximum = 255.0;
//
//   movingrescalefilter->SetOutputMinimum(desiredMinimum);
//   movingrescalefilter->SetOutputMaximum(desiredMaximum);
//   movingrescalefilter->UpdateLargestPossibleRegion();
//   fixedrescalefilter->SetOutputMinimum(desiredMinimum);
//   fixedrescalefilter->SetOutputMaximum(desiredMaximum);
//   fixedrescalefilter->UpdateLargestPossibleRegion();
//
//
//   // Histogram match the images
//   itk::HistogramMatchingImageFilter<itk::Image<float, 2>, itk::Image<float, 2>>::Pointer IntensityEqualizeFilter = itk::HistogramMatchingImageFilter<itk::Image<float, 2>, itk::Image<float, 2>>::New();
//
//   IntensityEqualizeFilter->SetReferenceImage(fixedrescalefilter->GetOutput());
//   IntensityEqualizeFilter->SetInput(movingrescalefilter->GetOutput());
//   IntensityEqualizeFilter->SetNumberOfHistogramLevels(100);
//   IntensityEqualizeFilter->SetNumberOfMatchPoints(15);
//   IntensityEqualizeFilter->ThresholdAtMeanIntensityOn();
//   IntensityEqualizeFilter->Update();
//
//   // // Set the images for registration filter
//   // registrationFilter->SetFixedImage(fixedrescalefilter->GetOutput());
//   // registrationFilter->SetMovingImage(IntensityEqualizeFilter->GetOutput());
//
//
//   //  WRITE RESULTS
//   itk::ImageFileWriter<itk::Image<float, 2>>::Pointer warpedImageWriter;
//   warpedImageWriter = itk::ImageFileWriter<itk::Image<float, 2>>::New();
//   warpedImageWriter->SetInput(IntensityEqualizeFilter->GetOutput());
//   warpedImageWriter->SetFileName("warpedMovingImage.mha");
//   try
//   {
//     warpedImageWriter->Update();
//   }
//   catch (itk::ExceptionObject & excp)
//   {
//     std::cerr << excp << std::endl;
//     return EXIT_FAILURE;
//   }
