//--------------------------------------------------------------------------//
//
//  ppmparser.h
//
//  Created by Didac Semente Fernandez on 09/04/2016
//
// Implementation of the ppmparser.h library.
//
//--------------------------------------------------------------------------//

//--------------------------------------------------------------------------//
// -- EXTERNAL LIBRARIES -------------------------------------------------- //
//--------------------------------------------------------------------------//

#include <ctype.h>
#include <mpi.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

//--------------------------------------------------------------------------//

#include "lib/ppmparser.h"

//--------------------------------------------------------------------------//
// -- MACRO DEFINITION -----------------------------------------------------//
//--------------------------------------------------------------------------//

#define TRUE 1
#define FALSE 0

//--------------------------------------------------------------------------//
// -- AUXILIARY METHODS ----------------------------------------------------//
//--------------------------------------------------------------------------//

void exchangeSections(DataBucket, int, int, int);

//--------------------------------------------------------------------------//
// -- LIBRARY IMPLEMENTATION ---------------------------------------------- //
//--------------------------------------------------------------------------//

long checkForRealloc(void **ptr, long allcBlcks, long actBlcks, size_t bSize,
    long incBlck) {

    long newAlloc = allcBlcks;
    void *temp = NULL;

    if(allcBlcks < actBlcks) {
        newAlloc = newAlloc + incBlck;
        if((temp = realloc(*ptr, newAlloc * bSize)) == NULL) {
            free(*ptr);
            return -1;
        } else {
            *ptr = temp;
        }
    }    

    return newAlloc;
}

int calcOffset(long size, int iwidth) {
    int offset;

    offset = size % 3;
    offset = offset + ((size - offset) % (iwidth * 3));

    return offset;
}

FILE* openFile(char* filename, char* mode) {
    return fopen(filename, mode);
}

int openMPIFile(MPI_File* mf, char* filename, int mode) {
    return MPI_File_open(MPI_COMM_WORLD, filename, mode, MPI_INFO_NULL, mf);
}

off_t fsize(const char *filename) {
    struct stat st; 

    if (stat(filename, &st) == 0) {
        return st.st_size;
    }

    return -1; 
}

ImageChunk* calculateChunkSections(FILE** f, ImageData img, int partitions) {

    intmax_t partsize = (img->rastersize / (intmax_t) partitions);
    intmax_t sizeleft = img->rastersize;

    ImageChunk *chunkLst;

    chunkLst = (ImageChunk*) malloc(sizeof(ImageChunk) * partitions);

    if(chunkLst == NULL) {
        return NULL;
    }

    for(int i = 0; i < partitions; i++) {
        chunkLst[i] = (ImageChunk) malloc(sizeof(struct imgchunk));
        if(chunkLst[i] == NULL) {
            return NULL;
        }
    }

    chunkLst[0]->start = img->headersize;
    chunkLst[0]->end = getAdjustedPoint(f, img->headersize + partsize);

    sizeleft = sizeleft - (chunkLst[0]->end - chunkLst[0]->start);

    for(int i = 1; i < partitions; i++) {
        partsize = sizeleft / (partitions - i);
        chunkLst[i]->start = chunkLst[i-1]->end;
        chunkLst[i]->end = getAdjustedPoint(f, chunkLst[i]->start + partsize);
        sizeleft = sizeleft - (chunkLst[i]->end - chunkLst[i]->start);
    }

    return chunkLst;

}

void freeChunkList(ImageChunk* chunkLst, int lsize) {
    for(int i = 0; i < lsize; i++) {
        free(chunkLst[i]);
    }
    free(chunkLst);
}

DataBucket* initializeBuckets(int nbuckets, long bsize) {

    DataBucket *buckets;

    buckets = (DataBucket*) malloc(sizeof(DataBucket) * nbuckets);

    for(int i = 0; i < nbuckets; i++) {
        buckets[i] = malloc(sizeof(struct databucket));
        buckets[i]->bsize = 0;
        buckets[i]->blckSize = bsize;
        buckets[i]->offset = 0;
        buckets[i]->data = calloc(sizeof(int), bsize);
    }

    return buckets;
}

void freeDataBuckets(DataBucket* buckets, int nbuckets) {

    for(int i = 0; i < nbuckets; i++) {
        free(buckets[i]->data);
        free(buckets[i]);
    }
    free(buckets);
}

void transferUnalignedRasters(int prank, int pnum, DataBucket bucket, 
    int iwidth) {

    int transData = 0L;
    long bsize = bucket->bsize;
    int *buff = NULL;
    void **temp = NULL;
    MPI_Status status;

    temp = (void**) malloc(sizeof(int*));

    if(prank == 0) {
        transData = calcOffset(bsize, iwidth);
        buff = (int*) malloc(sizeof(int) * transData);
        memcpy(&buff[0], &bucket->data[bsize-transData], 
            transData * sizeof(int));
        bucket->bsize = (bsize - transData);
        MPI_Send((void*) &transData, 1 , MPI_INT, prank+1, 1, 
            MPI_COMM_WORLD);
        MPI_Send((void*) buff, transData, MPI_INT, prank+1, 
            1, MPI_COMM_WORLD);
    } else {
        MPI_Recv(&transData, 1, MPI_INT, prank-1, MPI_ANY_TAG, 
            MPI_COMM_WORLD, &status);
        buff = (int*) malloc(sizeof(int) * transData);
        MPI_Recv(buff, transData, MPI_INT, prank-1, MPI_ANY_TAG, 
            MPI_COMM_WORLD, &status);
        bsize += transData;

        *temp = bucket->data;
        bucket->blckSize = checkForRealloc(temp, bucket->blckSize, bsize,
            sizeof(int), transData);
        bucket->data = *temp;

        *temp = NULL;

        memmove(&bucket->data[transData], &bucket->data[0], 
            bucket->bsize * sizeof(int));
        memcpy(&bucket->data[0], &buff[0], transData * sizeof(int));
        bucket->bsize = bsize;

        free(buff);

        buff = NULL;

        if(prank < pnum-1) {
            transData = calcOffset(bsize, iwidth);

            buff = (int*) malloc(sizeof(int) * transData);

            memcpy(&buff[0], &bucket->data[bsize-transData], 
                transData * sizeof(int));
            bucket->bsize -= transData;
            MPI_Send((void*) &transData, 1 , MPI_INT, prank+1, 1, 
                MPI_COMM_WORLD);
            MPI_Send((void*) buff, transData, MPI_INT, prank+1, 
                1, MPI_COMM_WORLD);
        }
    }

    free(buff);

}

void transferBorders(int pc, int parts, int prank, int pnum, 
    DataBucket bucket, int imgWidth, int halosize) {

    int transData = imgWidth * 3 * halosize;

    void **temp = NULL;

    if(pc == 0) { // First partition
        exchangeSections(bucket, transData, prank+1, FALSE);

    } else if(pc == (parts * pnum) - 1) { // Last partition
        exchangeSections(bucket, transData, prank-1, TRUE);

    } else { // Rest
        if(prank < pnum-1) {
            exchangeSections(bucket, transData, prank+1, FALSE);

            if(prank > 0) {
                exchangeSections(bucket, transData, prank-1, TRUE);

            }
        } else {
            exchangeSections(bucket, transData,  prank-1, TRUE);

            bucket->bsize += transData;

            temp = (void**) malloc(sizeof(int*));
            
            *temp = bucket->data;
            bucket->blckSize = checkForRealloc(temp, bucket->blckSize, 
                bucket->bsize, sizeof(int), bucket->bsize-bucket->blckSize);
            bucket->data = *temp;

            *temp = NULL;

            free(temp);
        }

    }
}

void exchangeSections(DataBucket bucket, int transData, int dst, int putUp) {

    long dataSize = 0L;
    int transBytes = (transData * sizeof(int));
    int *trBuff = NULL, *rcvBuff = NULL;
    void **temp = NULL;

    MPI_Request request;
    MPI_Status status;

    dataSize = bucket->bsize;

    temp = (void**) malloc(sizeof(int*));
    trBuff = (int*) malloc(transBytes);
    rcvBuff = (int*) malloc(transBytes);

    if(temp == NULL || trBuff == NULL || rcvBuff == NULL) {
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    memcpy(&trBuff[0], &bucket->data[dataSize - transData], transBytes);

    MPI_Isend((void*) trBuff, transData, MPI_INT, dst, 1, MPI_COMM_WORLD, 
        &request);

    MPI_Recv((void*) &rcvBuff[0], transData, MPI_INT, dst, MPI_ANY_TAG, 
        MPI_COMM_WORLD, &status);

    *temp = bucket->data;
    bucket->blckSize = checkForRealloc(temp, bucket->blckSize, 
        (dataSize + transData), sizeof(int), transData);
    bucket->data = *temp;

    *temp = NULL;

    if(putUp == TRUE) {
        memmove(&bucket->data[transData], &bucket->data[0], 
            bucket->bsize * sizeof(int));
        memcpy(&bucket->data[0], &rcvBuff[0], transBytes);
    } else {
        memcpy(&bucket->data[dataSize], &rcvBuff[0], transBytes);
    }

    bucket->bsize = dataSize + transData;    

    MPI_Wait(&request, &status);

    free(temp);
    free(rcvBuff);
    free(trBuff);
}

void adjustBucketContents(DataBucket *buckets, int prank, int pnum, 
    int iwidth, int halosize) {

    long bsize = 0L;
    int transData = 0;
    int rasterOff, rowOff;
    int *buff = NULL;
    MPI_Status status;

    if(prank == 0) {
        MPI_Recv(&transData, 1, MPI_INT, pnum-1, MPI_ANY_TAG, 
            MPI_COMM_WORLD, &status);
        buff = (int*) malloc(sizeof(int) * transData);
        MPI_Recv(buff, transData, MPI_INT, pnum-1, MPI_ANY_TAG, 
            MPI_COMM_WORLD, &status);
        memcpy(&buckets[0]->data[0], &buff[0], transData * sizeof(int));
        buckets[0]->bsize = transData;
        buckets[0]->offset = transData - (halosize * iwidth * 3); //WHY?
    } else if(prank == pnum-1) {
        bsize = buckets[0]->bsize;
        rasterOff = (int) (bsize % 3);
        rowOff = (int) ((bsize - rasterOff) % (iwidth * 3));
        transData = rasterOff + rowOff + (halosize * 6 * iwidth);
        buff = (int*) malloc(sizeof(int) * transData);
        memcpy(&buff[0], &buckets[0]->data[bsize-transData], 
            transData * sizeof(int));
        MPI_Send((void*) &transData, 1 , MPI_INT, 0, 1, 
            MPI_COMM_WORLD);
        MPI_Send((void*) buff, transData, MPI_INT, 0, 
            1, MPI_COMM_WORLD);
        buckets[0]->bsize = 0;
        buckets[0]->offset = 0;
    } else {
        buckets[0]->bsize = 0;
        buckets[0]->offset = 0;
    }

    if(buff != NULL) {
        free(buff); 
    }

}

void adjustProcessBucket(DataBucket *buckets, int iwidth, int halosize) {

    long bsize = 0L;
    int offsetData = 0;

    int rasterOff = 0, rowOff = 0;

    bsize = buckets[0]->bsize;
    rasterOff = (int) (bsize % 3);
    rowOff = (int) ((bsize - rasterOff) % (iwidth * 3));

    offsetData = rasterOff + rowOff + (halosize * 6 * iwidth);

    memmove(&buckets[0]->data[0], &buckets[0]->data[bsize-offsetData], 
        offsetData * sizeof(int));

    buckets[0]->offset = offsetData;
    buckets[0]->bsize = offsetData;
}

intmax_t getAdjustedPoint(FILE** f, intmax_t next) {
    fseek(*f, next, SEEK_SET);
    char c;
    do {
        c = fgetc(*f);
    } while(isdigit(c));
    return (ftell(*f) - 1);
}

// Open Image file and image struct initialization
ImageData parseFileHeader(char* nombre, FILE **fp, int partitions, int halo, 
    double incFactor) {
    char c, comment[300];
    int i;
    ImageData img = NULL;
    long chunk = 0L;

    if ((*fp = openFile(nombre, "r")) == NULL) {
        perror("Error: ");
    } else {
        // Memory allocation
        img = (ImageData) malloc(sizeof(struct imageppm));

        // Storing file size
        img->rastersize = (intmax_t) fsize(nombre);
 
        // Reading the first line: Magical Number "P3"
        fscanf(*fp, "%c%d ", &c, &(img->P));
        
        // Reading the image comment
        for(i = 0; (c = fgetc(*fp)) != '\n'; i++) {
            comment[i] = c;
        }
        comment[i] = '\0';
        // Allocating information for the image comment
        img->comment = (char*) calloc(strlen(comment)+1, sizeof(char));
        strcpy(img->comment, comment);
        // Reading image dimensions and color resolution
        fscanf(*fp, "%d %d %d\n", &img->width, &img->height, &img->maxcolor);

        img->headersize = ftell(*fp);
        img->rastersize = img->rastersize - img->headersize;

        chunk = img->width * (img->height / partitions);
        // We need to read halo extra rows.
        chunk = chunk + img->width * halo;

        chunk = (long) ((float) chunk * incFactor);

        img->rsize = img->gsize = img->bsize = chunk;

        img->blckSize = chunk;

        img->R = calloc(img->rsize, sizeof(int));
        img->G = calloc(img->gsize, sizeof(int));
        img->B = calloc(img->bsize, sizeof(int));
        if((img->R == NULL) || (img->G == NULL) || (img->B == NULL) ||
            (img->rastersize == -1)) {
            return NULL;
        }
    }

    return img;
}

// Duplicate the Image struct for the resulting image
ImageData duplicateImageData(ImageData src, int partitions, int halo, 
    double incFactor) {
    long chunk;
    // Struct memory allocation
    ImageData dst = (ImageData) malloc(sizeof(struct imageppm));

    // Copying the magic number
    dst->P = src->P;
    // Copying the string comment
    dst->comment = calloc(strlen(src->comment)+1, sizeof(char));
    strcpy(dst->comment, src->comment);
    // Copying image dimensions and color resolution
    dst->width = src->width;
    dst->height = src->height;
    dst->maxcolor = src->maxcolor;
    chunk = dst->width * (dst->height / partitions);
    // We need to read an extra row.
    chunk = chunk + src->width * halo;

    chunk = (long) ((float) chunk * incFactor);

    dst->rsize = src->rsize;
    dst->gsize = src->gsize;
    dst->bsize = src->bsize;

    dst->blckSize = src->blckSize;

    dst->R = calloc(chunk, sizeof(int));
    dst->G = calloc(chunk, sizeof(int));
    dst->B = calloc(chunk, sizeof(int));
    if((dst->R == NULL) || (dst->G == NULL) || (dst->B == NULL)) {
        return NULL;
    }

    return dst;
}

//--------------------------------------------------------------------------//
