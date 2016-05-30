//--------------------------------------------------------------------------//
//
//  convolution.c
//
//  Created by Josep Lluis Lerida on 11/03/2015
//  Modified by Didac Semente Fernandez on 04/04/2016
//
// This program calculates the convolution for PPM images.
// The program accepts an PPM image file, a text definition of the kernel 
// matrix and the PPM file for storing the convolution results.
// The program allows to define image partitions for processing larger 
// images (>500MB).
// The 2D image is represented by 1D vector for chanel R, G and B. 
// The convolution is applied to each chanel separately.
//
//--------------------------------------------------------------------------//

//--------------------------------------------------------------------------//
// -- EXTERNAL LIBRARIES -------------------------------------------------- //
//--------------------------------------------------------------------------//

#include <ctype.h>
#include <mpi.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <unistd.h>

//--------------------------------------------------------------------------//

#include "lib/convolution.h"

//--------------------------------------------------------------------------//
// -- MACRO DEFINITION -----------------------------------------------------//
//--------------------------------------------------------------------------//

#define F_MICROS_IN_SECOND 1000000.0

#define TRUE 1
#define FALSE 0

#define REALLOC_MARGIN 10
#define INCREASE_FACTOR 100

//--------------------------------------------------------------------------//
// -- AUXILIARY METHODS ----------------------------------------------------//
//--------------------------------------------------------------------------//

int validateParameters(char**);
double calculateExtraSize(int partitions);
long rebuildImage(ImageData, DataBucket*);
long calcRasterWriteAmount(int*, long, long);
long calculateWriteAmount(ImageData, int, int, int);

//--------------------------------------------------------------------------//
// -- LIBRARY IMPLEMENTATION ---------------------------------------------- //
//--------------------------------------------------------------------------//

// Read the corresponding chunk from the source Image
int readChunk(MPI_File* mfp, intmax_t *offset, intmax_t *limit, 
    DataBucket bucket) {

    intmax_t pos = *offset;
    int value = 0, mult = 10;
    int newValue = FALSE;
    int increase = INCREASE_FACTOR;
    long k = bucket->offset, bucketBlockSize, i = 0;
    char c;

    char *cbuff = NULL;

    MPI_Status status;

    int **temp = NULL;

    temp = (int**) malloc(sizeof(int*)); // Avoid breaking strict aliasing

    cbuff = (char*) malloc(sizeof(char) * (*limit - *offset + 1));

    MPI_File_set_view(*mfp, *offset, MPI_CHAR, MPI_CHAR, 
            "native", MPI_INFO_NULL);

    MPI_File_read(*mfp, &cbuff[0], (*limit - *offset + 1), MPI_CHAR, &status);

    while(pos <= *limit) { 
        c = cbuff[i];
        if(isdigit(c)) {
            value = (value * mult) + (c - '0');
            newValue = TRUE;
        } else if(newValue) {
            bucket->data[k] = value;
            value = 0;
            newValue = FALSE;
            k++;
            // CHECKING IF WE ARE ABOUT TO FILL THE BUCKET
            *temp = bucket->data; 
            bucketBlockSize = bucket->blckSize;
            bucket->blckSize = checkForRealloc((void**) temp, 
                bucket->blckSize, (k + REALLOC_MARGIN), 
                sizeof(bucket->data[0]), increase);
            bucket->data = *temp;
            if(bucketBlockSize < bucket->blckSize) {
                increase *= 2;
            } else if(bucket->blckSize == -1) {
                perror("Error: ");
                return -1;
            }
        }     
        pos += 1;
        i += 1;
    }

    bucket->bsize = k;

    free(temp);
    free(cbuff);

    return 0;
}

// Duplication of the just readed source chunk 
// to the destiny image struct chunk
void* duplicateImageChunk(ImageData src, ImageData dst) {

    int** temp = NULL;
    long blckInc = (src->blckSize - dst->blckSize);

    temp = (int**) malloc(sizeof(int*)); // Avoid breaking strcit aliasing

    *temp = dst->R;
    checkForRealloc((void**) temp, dst->blckSize, src->blckSize, 
        sizeof(dst->R[0]), blckInc);
    dst->R = *temp;

    *temp = dst->G;
    checkForRealloc((void**) temp, dst->blckSize, src->blckSize, 
        sizeof(dst->G[0]), blckInc);
    dst->G = *temp;

    *temp = dst->B;
    dst->blckSize = checkForRealloc((void**) temp, dst->blckSize, 
        src->blckSize, sizeof(dst->B[0]), blckInc);
    dst->B = *temp;

    free(temp);

    if(dst->blckSize == -1) {
        return NULL;
    }
    
    if(memcpy((void*) dst->R, (void*) src->R, 
        dst->rsize * sizeof(dst->R[0])) == NULL) {
        return NULL;
    }

    if(memcpy((void*) dst->G, (void*) src->G, 
        dst->gsize * sizeof(dst->G[0])) == NULL) {
        return NULL;
    }

    return memcpy((void*) dst->B, (void*) src->B, 
        dst->bsize * sizeof(dst->B[0]));
}

// Open kernel file and reading kernel matrix. 
// The kernel matrix 2D is stored in 1D format.
KernelData readKernel(char* fileName) {
    FILE *fp;
    int ksize = 0;
    KernelData kern = NULL;
    
    // Opening the kernel file
    if((fp = openFile(fileName, "r")) == NULL) {
        perror("Error: ");
    } else {
        // Memory allocation
        kern = (KernelData) malloc(sizeof(struct structkernel));
        
        // Reading kernel matrix dimensions
        fscanf(fp, "%d,%d,", &kern->kernelX, &kern->kernelY);
        ksize = (kern->kernelX * kern->kernelY);
        kern->vkern = (float*) malloc(ksize * sizeof(float));
        
        // Reading kernel matrix values
        for(int i = 0; i < ksize; i++) {
            fscanf(fp, "%f,", &kern->vkern[i]);
        }

        fclose(fp);
    }

    return kern;
}

// Open the image file with the convolution results
int initfilestore(ImageData img, FILE** fp, char* fileName, long *position) {
    // File with the resulting image is created
    if((*fp = openFile(fileName, "w")) == NULL) {
        perror("Error: ");
        return -1;
    } 
    
    // Writing image header
    fprintf(*fp, "P%d\n%s\n%d %d\n%d\n", img->P, img->comment, img->width,
        img->height, img->maxcolor);
    *position = ftell(*fp);
    return 0;
}

// Writing the image chunk to the resulting file.
int savingChunk(ImageData img, MPI_File *mfp, long *offset, long dataOffst, 
    long count, long writeAmount){

    long end = count + dataOffst;
    MPI_Status status;

    char *obuff = NULL;

    obuff = malloc(sizeof(char) * (writeAmount + 1));

    for(long i = dataOffst, k = 0; i < end; i++) {
        k += sprintf(&obuff[k], "%d ", img->R[i]);
        k += sprintf(&obuff[k], "%d ", img->G[i]);
        k += sprintf(&obuff[k], "%d\n", img->B[i]);
    }

    MPI_File_set_view(*mfp, *offset, MPI_CHAR, MPI_CHAR, "native", 
        MPI_INFO_NULL);

    // Writing image partition
    MPI_File_write(*mfp, (void*) &obuff[0], writeAmount, MPI_CHAR, &status);

    free(obuff);

    return 0;
}

// This function frees the space allocated for the image structure.
void freeImagestructure(ImageData *src) {
    
    free((*src)->comment);
    free((*src)->R);
    free((*src)->G);
    free((*src)->B);
    free(*src);
}

//--------------------------------------------------------------------------//
// 2D convolution
// 2D data are usually stored in computer memory as contiguous 1D array.
// So, we are using 1D array for 2D data.
// 2D convolution assumes the kernel is center originated, which means, if
// kernel size 3 then, k[-1], k[0], k[1]. The middle of index is always 0.
// The following programming logics are somewhat complicated because of using
// pointer indexing in order to minimize the number of multiplications.
//
//
// signed integer (32bit) version:
//--------------------------------------------------------------------------//
int convolve2D(int* in, int* out, int dataSizeX, int dataSizeY, int dataOff,
               float* kernel, int kernelSizeX, int kernelSizeY) {

    int *inPtr = NULL, *inPtr2 = NULL, *outPtr = NULL;
    float *kPtr = NULL;
    int kCenterX, kCenterY;
    long rowMin, rowMax;                 // to check boundary of input array
    long colMin, colMax;                //
    float sum;                         // temp accumulation buffer
    
    // Parameter validatin
    if(!in || !out || !kernel || dataSizeX <= 0 || kernelSizeX <= 0) { 
        return -1;
    }
    
    // Find centeral position of kernel (half of kernel size)
    kCenterX = (int) kernelSizeX / 2;
    kCenterY = (int) kernelSizeY / 2;
    
    // init working  pointers
    // note that  it is shifted (kCenterX, kCenterY),
    inPtr = inPtr2 = &in[(dataSizeX * kCenterY) + kCenterX];
    outPtr = out;
    kPtr = kernel;
    
    // start convolution
    // number of rows
    for(register int i = 0; i < dataSizeY; ++i) {
        // compute the range of convolution, the current row of kernel 
        // should be between these
        rowMax = i + kCenterY;
        rowMin = i - dataSizeY + kCenterY;
        
        // number of columns
        for(register int j = 0; j < dataSizeX; ++j) {
            // compute the range of convolution, the current column of kernel 
            // should be between these
            colMax = j + kCenterX;
            colMin = j - dataSizeX + kCenterX;
            
            sum = 0.0f;                        // set to 0 before accumulate
            
            // flip the kernel and traverse all the kernel values
            // multiply each kernel value with underlying input data
            // kernel rows
            for(register int m = 0; m < kernelSizeY; ++m) {
                // check if the index is out of bound of input array
                if(m <= rowMax && m > rowMin) {
                    for(register int n = 0; n < kernelSizeX; ++n) {
                        // check the boundary of array
                        if(n <= colMax && n > colMin) {
                            sum += *(inPtr - n) * (*kPtr);
                        }
                        
                        ++kPtr;// next kernel
                    }
                } else {
                    // out of bound, move to next row of kernel
                    kPtr += kernelSizeX; 
                }
                
                // move input data 1 raw up
                inPtr -= dataSizeX;   
            }
            
            // convert integer number
            if(sum >= 0.0f) { 
                *outPtr = (int)(sum + 0.5f);
            } else  { // For using with image editors like GIMP or others...
                *outPtr = (int)(sum - 0.5f);
            }
            
            kPtr = kernel; // reset kernel to (0,0)
            inPtr = ++inPtr2; // next input
            ++outPtr; // next output
        }
    }
    
    return 0;
}

//--------------------------------------------------------------------------//
// -- AUXILIARY METHODS IMPLEMENTATION ------------------------------------ //
//--------------------------------------------------------------------------//

int validateParameters(char **args) {
    if(access(args[1], F_OK)) {
        perror("Input image error");
        return -1;
    } else if(access(args[2], F_OK)) {
        perror("Kernel file error");
        return -1;
    } else if(atoi(args[4]) < 1) {
        printf("Partition number error: value less than 1\n");
        return -1;
    }
    return 0;
}

double calculateExtraSize(int partitions) {
    double x = (double) partitions;
    return (x / (15 + 3*x)) - 0.058f;
}

// Method used to fill the ImageData structure using the data found in the
// DataBucket list.
long rebuildImage(ImageData img, DataBucket *bucks) {
    long r, g, b, tsize;
    long rasterR, rasterG, rasterB;
    long increaseR, increaseG, increaseB;
    long memR, memG, memB;
    int flip, **temp = NULL;

    r = g = b = 0L;
    flip = 0;
    increaseR = increaseG = increaseB = INCREASE_FACTOR * 10;
    memR = memG = memB = 0L;
    
    temp = (int**) malloc(sizeof(int*)); // Avoid breaking strict aliasing

    for(int i = 0; i < 1; i++) {
        for(int j = 0; j < bucks[i]->bsize; j++) {
            switch(flip) {
                case 0:
                    img->R[r] = bucks[i]->data[j];
                    r++;
                    rasterR = img->blckSize;
                    *temp = img->R;
                    memR = checkForRealloc((void**) temp, img->blckSize, 
                        r + REALLOC_MARGIN, sizeof(int), increaseR);
                    img->R = *temp;
                    if(rasterR < memR) {
                        increaseR *= 2;
                    }
                    break;
                case 1:
                    img->G[g] = bucks[i]->data[j];
                    g++;
                    rasterG = img->blckSize;
                    *temp = img->G;
                    memG = checkForRealloc((void**) temp, img->blckSize, 
                        g + REALLOC_MARGIN, sizeof(int), increaseG);
                    img->G = *temp;
                    if(rasterG < memG) {
                        increaseG *= 2;
                    }
                    break;
                case 2:
                    img->B[b] = bucks[i]->data[j];
                    b++;
                    rasterB = img->blckSize;
                    *temp = img->B;
                    memB = checkForRealloc((void**) temp, img->blckSize, 
                        b + REALLOC_MARGIN, sizeof(int), increaseB);
                    img->B = *temp;
                    if(rasterB < memB) {
                        increaseB *= 2;
                        img->blckSize = memB;
                    }
                    break;
            }
            *temp = NULL;
            flip = (flip + 1) % 3;
        }
        bucks[i]->offset = 0;
    }

    *temp = NULL;

    free(temp);

    tsize = (r + g + b);

    // Check for unaligned rasters
    // Either 1 Blue is missing from the image or
    // both 1 Green and 1 Blue.
    switch(tsize % 3) {
        case 0:
            break;
        case 2:
            bucks[0]->offset += 1;
            tsize -= 1;
        case 1:
            bucks[0]->offset += 1;
            tsize -= 1;
            break;
    }

    return (tsize / 3);
}

long calcRasterWriteAmount(int *raster, long offset, long end) {

    long wAmount = 0L;

    char value[10];

    while(offset < end) {
        wAmount += sprintf(&value[0], "%d", raster[offset]);
        wAmount += 1; // Seperator character = 1 byte
        offset++;
    }

    return wAmount;
}

long calculateWriteAmount(ImageData img, int offset, int chunksize, 
    int imgWidth) {

    long i = offset * imgWidth, 
         end = (chunksize * imgWidth) + i;

    long writeAmount = 0L;

    writeAmount += calcRasterWriteAmount(img->R, i, end);
    writeAmount += calcRasterWriteAmount(img->G, i, end);
    writeAmount += calcRasterWriteAmount(img->B, i ,end);

    return writeAmount;
}

//--------------------------------------------------------------------------//
// - MAIN METHOD -----------------------------------------------------------//
//--------------------------------------------------------------------------//

int main(int argc, char **argv) {

    int c, offset, pc;
    int prank, pnum;
    int partitions, effectivePart, halo, haloSize;
    int imgWidth, imgHeight;
    int convOffset, convSize;
    long *writeOffs = NULL;
    long totalWritten = 0L, writeSize = 0L;

    long bposition, position, chunkSize, iterSize, bucketSize;
    
    double start, tstart, tend, tread, tcopy, tconv, tstore, treadk;
    float extraSizeFactor;

    char *sourceFile, *outFile, *kernFile;
    char cwd[1024];

    FILE *fpsrc, *fpdst;

    MPI_File *mfpsrc, *mfpdst;

    ImageData source, output;

    KernelData kern;

    ImageChunk *chunkLst;

    DataBucket *buckets;

    c = offset = 0;
    position = 0L;
    tstart = tend = tread = tcopy = tconv = tstore = treadk = 0.0;
    sourceFile = outFile = kernFile = NULL;
    fpsrc = fpdst = NULL;
    mfpsrc = mfpdst = NULL;
    source = output = NULL;
    kern = NULL;

    extraSizeFactor = 1.0f;
    
    if(argc != 5) {
        printf("Usage: %s <image-file> <kernel-file> <result-file> "
            "<partitions> \n\n", argv[0]);
        
        printf("- image_file : source image path (*.ppm)\n");
        printf("- kernel_file: kernel path (text file with 1D "
            "kernel matrix)\n");
        printf("- result_file: result image path (*.ppm)\n");
        printf("- partitions : Image partitions\n");
        return -1;
    }

    if(validateParameters(argv) == -1) {
        return -1;
    }

    MPI_Init(&argc, &argv);

    start = MPI_Wtime();
    tstart = start;

    MPI_Comm_size(MPI_COMM_WORLD, &pnum);
    MPI_Comm_rank(MPI_COMM_WORLD, &prank);

    //Storing parameters
    sourceFile = argv[1];
    kernFile = argv[2];
    outFile = argv[3];
    partitions = atoi(argv[4]);

    effectivePart = partitions * pnum;

    writeOffs = (long*) malloc(sizeof(long) * pnum);

    getcwd(cwd, sizeof(cwd));

    // Opening files
    mfpsrc = (MPI_File*) malloc(sizeof(MPI_File));
    mfpdst = (MPI_File*) malloc(sizeof(MPI_File));

    openMPIFile(mfpsrc, sourceFile, MPI_MODE_RDONLY);
    openMPIFile(mfpdst, outFile, MPI_MODE_WRONLY | MPI_MODE_CREATE);
    
    // READING IMAGE HEADERS, KERNEL Matrix, DUPLICATE IMAGE DATA, 
    // OPEN RESULTING IMAGE FILE

    // Reading kernel matrix
    start = MPI_Wtime();
    if ((kern = readKernel(kernFile)) == NULL) {
        return -1;
    }

    // The matrix kernel defines the halo size to use with the image. 
    // The halo is zero when the image is not partitioned.
    if (effectivePart == 1) {
        halo = 0;
    } else { 
        halo = kern->kernelY;
    }

    treadk = MPI_Wtime() - start;

    // Reading Image Header. Image properties: Magical number, comment, 
    // size and color resolution.
    start = MPI_Wtime();

    // Calculating extra size for memory assignment in order to avoid
    // calling realloc further in the execution
    extraSizeFactor = extraSizeFactor + calculateExtraSize(effectivePart);

    // Memory allocation based on number of partitions and halo size.
    if((source = parseFileHeader(sourceFile, &fpsrc, effectivePart, 
        halo, extraSizeFactor)) == NULL) {
        return -1;
    }

    imgWidth = source->width;
    imgHeight = source->height;

    bposition = source->headersize;
    totalWritten = bposition;

    tread = tread + (MPI_Wtime() - start);
    
    // Duplicate the image struct.
    start = MPI_Wtime();
    if ((output = duplicateImageData(source, effectivePart, halo, 
        extraSizeFactor)) == NULL) {
        return -1;
    }
    tcopy = tcopy + (MPI_Wtime() - start);
    
    // Initialize Image output file. Open the file and store the image header
    start = MPI_Wtime();
    if(prank == 0) { 
        if (initfilestore(output, &fpdst, outFile, &position) != 0) {
            perror("Error: ");
            return -1;
        }
        fclose(fpdst);
        
    }

    tstore = tstore + (MPI_Wtime() - start);

    bucketSize = (imgWidth * imgHeight * 3) / effectivePart;
    bucketSize = bucketSize + (imgWidth * halo);

    bucketSize = (long) ((float) bucketSize * extraSizeFactor);

    chunkLst = calculateChunkSections(&fpsrc, source, effectivePart);

    fclose(fpsrc);

    if ((buckets = initializeBuckets(1, bucketSize)) == NULL) {
        perror("Error: ");
        return -1;
    }

    //----------------------------------------------------------------------//
    // CHUNK PROCESSING LOOP
    //----------------------------------------------------------------------//

    while (c < partitions) {

        pc = (pnum * c) + prank;

        // Reading chunk.
        start = MPI_Wtime();

        if (readChunk(mfpsrc, &(chunkLst[pc]->start), 
            &(chunkLst[pc]->end), buckets[0])) {
             return -1;
        }

        tread = tread + (MPI_Wtime() - start);

        if(pnum > 1) {
            transferUnalignedRasters(prank, pnum, buckets[0], imgWidth);
        }

        haloSize = (halo / 2);

        if(pnum > 1) {
            transferBorders(pc, partitions, prank, pnum, buckets[0], 
                imgWidth, haloSize);
        }

        // Copying data from the DataBucket into the ImageData arrays
        //if(gPrank == 0 && c < 1)
        iterSize = rebuildImage(source, buckets);

        // Discarding incomplete row.
        convOffset = (iterSize % imgWidth);
        convSize = iterSize - convOffset;

        // Rows to convolve needs to be bigger than kernel size, either way
        // there'll be problems in pixel alignment.
        if(pc < (effectivePart-1)) {
            chunkSize = (convSize / imgWidth) - haloSize;
            if(pc == 0) {
                offset = 0;
            } else {
                offset = haloSize;
            }
        } else {
            chunkSize = (convSize / imgWidth);
            offset = haloSize;
        }
        
        // Duplicate the image chunk
        start = MPI_Wtime();
        if (duplicateImageChunk(source, output) == NULL) {
            perror("Error: ");
            return -1;
        }
        tcopy = tcopy + (MPI_Wtime() - start);
        
        //------------------------------------------------------------------//
        // - CHUNK CONVOLUTION ---------------------------------------------//
        //------------------------------------------------------------------//
        start = MPI_Wtime();

        convolve2D(source->R, output->R, imgWidth, chunkSize, 
        	offset, kern->vkern, kern->kernelX, kern->kernelY);
        convolve2D(source->G, output->G, imgWidth, chunkSize, 
        	offset, kern->vkern, kern->kernelX, kern->kernelY);
        convolve2D(source->B, output->B, imgWidth, chunkSize, 
        	offset, kern->vkern, kern->kernelX, kern->kernelY);
        
        tconv = MPI_Wtime() - start;

        //------------------------------------------------------------------//
        // - CHUNK SAVING --------------------------------------------------//
        //------------------------------------------------------------------//

        if(pc > 0) {
            offset = haloSize;
            if(pc < (effectivePart - 1)) {
                chunkSize = (convSize / imgWidth) - (haloSize * 2);
            } else {
                chunkSize = (convSize / imgWidth) - haloSize;
            }
        } else {
            offset = 0;
            chunkSize = (convSize / imgWidth) - haloSize;
        }

        writeSize = calculateWriteAmount(output, offset, chunkSize, 
            imgWidth);

        MPI_Allgather((void*) &writeSize, 1, MPI_LONG, (void*) &writeOffs[0], 
            1, MPI_LONG, MPI_COMM_WORLD);

        position = totalWritten;

        for(int i = 0; i < pnum; i++) {
            if(i < prank) {
                position = position + writeOffs[i];
            }
            totalWritten = totalWritten + writeOffs[i];
        }

        start = MPI_Wtime();

        if (savingChunk(output, mfpdst, &position, (offset * imgWidth), 
                        (chunkSize * imgWidth), writeSize)) {
            perror("Error: ");
            return -1;
        }

        tstore = tstore + (MPI_Wtime() - start);

        // Moving previously discarded pixels to the beginning of the bucket
        // for the next iteration
        if(c < partitions-1) {
            if(pnum > 1) {
                adjustBucketContents(buckets, prank, pnum, imgWidth, 
                    haloSize);
            } else {
                adjustProcessBucket(buckets, imgWidth, haloSize);
            }
        }

        c++;
    }

    MPI_File_close(mfpsrc);
    MPI_File_close(mfpdst);
    
    tend = MPI_Wtime();

    if(prank == 0) {
    
        /*printf("-----------------------------------\n");
        printf("|       TYPE SIZES (BYTES)        |\n");
        printf("-----------------------------------\n");
        printf("Size of short: ----> %ld\n", sizeof(short));
        printf("Size of int: ------> %ld\n", sizeof(int));
        printf("Size of long: -----> %ld\n", sizeof(long));
        printf("Size of intmax_t: -> %ld\n", sizeof(intmax_t));
        printf("Size of size_t: ---> %ld\n", sizeof(size_t));
        printf("Size of float: ----> %ld\n", sizeof(float));
        printf("Size of double: ---> %ld\n", sizeof(double));
        printf("-----------------------------------\n");
        printf("|          IMAGE INFO             |\n");
        printf("-----------------------------------\n");
        printf("Working directory: %s\n", cwd);
        printf("File path: %s\n", sourceFile);
        printf("File output: %s\n", outFile);
        printf("Header size (bytes): %ld\n", source->headersize);
        printf("Raster size (bytes): %jd\n", source->rastersize);
        printf("ISizeX : %d\n", imgWidth);
        printf("ISizeY : %d\n", imgHeight);
        printf("kSizeX : %d\n", kern->kernelX);
        printf("kSizeY : %d\n", kern->kernelY);
        printf("-----------------------------------\n");
        printf("|         EXECUTION TIMES         |\n");
        printf("-----------------------------------\n");
        printf("%.6lfs elapsed in reading image file.\n", tread);
        printf("%.6lfs elapsed in copying image structure.\n", tcopy);
        printf("%.6lfs elapsed in reading kernel matrix.\n", treadk);
        printf("%.6lfs elapsed computing the convolution.\n", tconv);
        printf("%.6lfs elapsed in writing the resulting image.\n", tstore);
        printf("-----------------------------------\n");
        printf("%.6lfs elapsed in total.\n", tend-tstart);
        printf("-----------------------------------\n");*/
        printf("%s %s %d %.3lf\n", sourceFile, kernFile, pnum, tend-tstart);
    }

    //----------------------------------------------------------------------//
    // - MEMORY CLEANING  --------------------------------------------------//
    //----------------------------------------------------------------------//

    freeImagestructure(&source);
    freeImagestructure(&output);
    freeDataBuckets(buckets, 1);
    freeChunkList(chunkLst, effectivePart);
    free(kern->vkern);
    free(kern);
    free(mfpsrc);
    free(mfpdst);
    free(writeOffs);

    //----------------------------------------------------------------------//

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();

    return 0;
}

//--------------------------------------------------------------------------//
