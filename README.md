# CCAugmentation

[![Build Package](https://github.com/pijuszczyk/CCAugmentation/actions/workflows/build-package.yml/badge.svg?branch=master)](https://github.com/pijuszczyk/CCAugmentation/actions/workflows/build-package.yml)

Data preprocessing and augmentation framework that is designed for working with crowd counting datasets.
It supports multitude of simple as well as advanced transformations
and is based on pipelines which allow a flexible flow of data between loaders, transformations and outputs.
Deep learning framework-independent, though works best with PyTorch.

You can see the documentation [here](https://pijuszczyk.github.io/CCAugmentation/)

## Current capabilities

Each data preprocessing procedure is defined in form of a pipeline that consists of a data loader and
a list of operations to sequentially execute on the data. Each of the operations may be of the following types:
- **Transformation** - Returns transformed data on output, does not have side effects
- **Output** - Returns unmodified data on output, has side effects that, for example, write data to files
- **Operation** - Performs any other functions, not qualifying for any of the aforementioned types

Available transformations are:
- **Crop**
- **Scale**
- **Downscale**
- **Rotate**
- **StandardizeSize**
- **Normalize**
- **NormalizeDensityMap**
- **FlipLR**
- **ToGrayscale**
- **LambdaTransformation**
- **Cutout**
- **Shearing**
- **Blur**
- **BlurCutout**
- **DistortPerspective**
- **ChangeSaturation**
- **Mixup**

Available outputs are:
- **Demonstrate**
- **SaveImagesToFiles**
- **SaveImagesToBinaryFile**
- **SaveDensityMapsToCSVFiles**
- **SaveDensityMapsToBinaryFile**

Available operations are:
- **Duplicate**
- **Dropout**
- **RandomArgs**
- **OptimizeBatch**

Available loaders are:
- **BasicImageFileLoader**
- **ImageFileLoader**
- **BasicGTPointsMatFileLoader**
- **GTPointsMatFileLoader**
- **BasicDensityMapCSVFileLoader**
- **DensityMapCSVFileLoader**
- **VariableLoader**
- **ConcatenatingLoader**
- **CombinedLoader**

You can also use builtin integrations for:
- **PyTorch**
- **ShangaiTech dataset**
- **NWPU dataset**

For more information about the specific topics, please refer to the related comments in the code.

## How to use

Loading the data from ShanghaiTech dataset and taking crops with 1/4 size:

    import CCAugmentation as cca
    import CCAugmentation as ccat
    
    
    train_data_pipeline = cca.Pipeline(
        cca.integrations.datasets.SHHLoader("/data/ShanghaiTech/", "train", "B"),
        [
            ccat.Crop(None, None, 1/4, 1/4)
        ]
    )
    
    train_img, train_dm = train_data_pipeline.execute_collect()
    # you can also use execute_generate() to create a generator
    
    print(len(train_img), len(train_dm))

To see more examples of usage, please see our [experiment environment repository](https://github.com/m-konopka/CCAugmentation-Experiments-Env).

You can also preview the documentation for this project in a browser, using *pdoc3* docs generation. Here's how you can easily do this:

    $ pip install pdoc3
    $ pdoc --html --output-dir docs CCAugmentation
