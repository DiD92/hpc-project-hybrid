//--------------------------------------------------------------------------//

#ifndef __PPM_PARSER_H__
#define __PPM_PARSER_H__

//--------------------------------------------------------------------------//
//
//  ppmparser.h
//
//  Created by Didac Semente Fernandez on 09/04/2016
//
// Parser used to read PPM files compliant with the PPM format standard,
// standard source: http://netpbm.sourceforge.net/doc/ppm.html
//
//--------------------------------------------------------------------------//

//--------------------------------------------------------------------------//
// -- EXTERNAL LIBRARIES -------------------------------------------------- //
//--------------------------------------------------------------------------//

#include <stdio.h>
#include <stdint.h>
#include <sys/types.h>

//--------------------------------------------------------------------------//
// -- TYPE DEFINITIONS ---------------------------------------------------- //
//--------------------------------------------------------------------------//

struct imageppm {
    long headersize;
    intmax_t rastersize;
    int height;
    int width;
    char *comment;
    int maxcolor;
    int P;

    long blckSize;

    long rsize;
    long bsize;
    long gsize;

    int *R;
    int *G;
    int *B;
};

typedef struct imageppm* ImageData;

//--------------------------------------------------------------------------//

struct structkernel {
    int kernelX;
    int kernelY;
    float *vkern;
};

typedef struct structkernel* KernelData;

//--------------------------------------------------------------------------//

struct imgchunk {
    intmax_t start;
    intmax_t end;
};

typedef struct imgchunk* ImageChunk;

//--------------------------------------------------------------------------//

struct databucket {

    long bsize;
    long blckSize;

    int offset;
    int *data;
};

typedef struct databucket* DataBucket;

//--------------------------------------------------------------------------//
// -- METHOD DEFINITION --------------------------------------------------- //
//--------------------------------------------------------------------------//

long checkForRealloc(void**, long, long, size_t, long);
int calcOffset(long, int);
FILE* openFile(char*, char*);
int openMPIFile(MPI_File*, char*, int);
off_t fsize(const char*);
ImageChunk* calculateChunkSections(FILE**, ImageData, int);
void freeChunkList(ImageChunk*, int);
DataBucket* initializeBuckets(int, long);
void freeDataBuckets(DataBucket*, int);
void transferUnalignedRasters(int, int, DataBucket, int);
void transferBorders(int, int, int, int, DataBucket, int, int);
void adjustBucketContents(DataBucket*, int, int, int, int);
void adjustProcessBucket(DataBucket*, int, int);
intmax_t getAdjustedPoint(FILE** f, intmax_t next);
ImageData parseFileHeader(char*, FILE**, int, int, double);
ImageData duplicateImageData(ImageData, int, int, double);

//--------------------------------------------------------------------------//

#endif

//--------------------------------------------------------------------------//
