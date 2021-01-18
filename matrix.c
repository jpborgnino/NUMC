#include "matrix.h"
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

// Include SSE intrinsics
#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
#include <immintrin.h>
#include <x86intrin.h>
#endif

/* Below are some intel intrinsics that might be useful
 * void _mm256_storeu_pd (double * mem_addr, __m256d a)
 * __m256d _mm256_set1_pd (double a)
 * __m256d _mm256_set_pd (double e3, double e2, double e1, double e0)
 * __m256d _mm256_loadu_pd (double const * mem_addr)
 * __m256d _mm256_add_pd (__m256d a, __m256d b)
 * __m256d _mm256_sub_pd (__m256d a, __m256d b)
 * __m256d _mm256_fmadd_pd (__m256d a, __m256d b, __m256d c)
 * __m256d _mm256_mul_pd (__m256d a, __m256d b)
 * __m256d _mm256_cmp_pd (__m256d a, __m256d b, const int imm8)
 * __m256d _mm256_and_pd (__m256d a, __m256d b)
 * __m256d _mm256_max_pd (__m256d a, __m256d b)
*/

/* Generates a random double between low and high */
double rand_double(double low, double high) {
    double range = (high - low);
    double div = RAND_MAX / range;
    return low + (rand() / div);
}

/* Generates a random matrix */
void rand_matrix(matrix *result, unsigned int seed, double low, double high) {
    srand(seed);
    for (int i = 0; i < result->rows; i++) {
        for (int j = 0; j < result->cols; j++) {
            set(result, i, j, rand_double(low, high));
        }
    }
}

/*
 * Allocates space for a matrix struct pointed to by the double pointer mat with
 * `rows` rows and `cols` columns. You should also allocate memory for the data array
 * and initialize all entries to be zeros. `parent` should be set to NULL to indicate that
 * this matrix is not a slice. You should also set `ref_cnt` to 1.
 * You should return -1 if either `rows` or `cols` or both have invalid values, or if any
 * call to allocate memory in this function fails. Return 0 upon success.
 */
int allocate_matrix(matrix **mat, int rows, int cols) {
    if ((rows < 1) | (cols < 1)){
        PyErr_SetString(PyExc_TypeError, "Dimensions wrong");
       return -1;
   }
   if (mat == NULL){
       return -1;
   }
   *mat = malloc(sizeof(matrix));
   if ((*mat) == NULL){
       PyErr_SetString(PyExc_RuntimeError, "Failed to add");
       return -1;
   }
   (*mat)->rows = rows;
   (*mat)->cols = cols;
   (*mat)->parent = NULL;
   (*mat)->ref_cnt = 1;
   (*mat)->data = NULL;
   double *tempd = malloc(sizeof(double) * (rows * cols));
 
   if ( tempd == NULL){
       return -1;
   }
   int len = rows * cols;
   
   if(len < 50){
   
   for (unsigned int i = 0; i < len; i++ ){
       tempd[i] = 0.0;
   
   }
   }
   else
  {
    __m256d tempload = _mm256_set1_pd(0.0);  
   int i;
   int t;
   #pragma omp parallel shared(tempload) private(i,t)
   {

   #pragma omp for schedule(static) nowait
      for (unsigned int i = 0; i < (len / 4)*4; i +=4){
          _mm256_storeu_pd( tempd+i, tempload);
  }
  
  #pragma omp for schedule(static)
  for (unsigned int t = (len/ 4) * 4; t < len; t++){
      tempd[t] = 0.0;
  }

   
   
   }
  }

   
   (*mat)->data = tempd;


   return 0;  
}


/*
 * Allocates space for a matrix struct pointed to by `mat` with `rows` rows and `cols` columns.
 * Its data should point to the `offset`th entry of `from`'s data (you do not need to allocate memory)
 * for the data field. `parent` should be set to `from` to indicate this matrix is a slice of `from`.
 * You should return -1 if either `rows` or `cols` or both are non-positive or if any
 * call to allocate memory in this function fails. Return 0 upon success.
 */
int allocate_matrix_ref(matrix **mat, matrix *from, int offset, int rows, int cols) {
    if ((rows < 1) | (cols < 1) | (offset < 0)){
       PyErr_SetString(PyExc_TypeError, "Dimensions wrong");
       return -1;
   }
   if (from == NULL){
       return -1;
   }
   if (from->parent != NULL){
       return -1;
   }
 
   if (mat == NULL){
       return -1;
   }
   *mat = malloc(sizeof(matrix));
   if ((*mat) == NULL){
       return -1;
   }
   (*mat)->rows = rows;
   (*mat)->cols = cols;
   (*mat)->parent = from;
   (*mat)->ref_cnt = 1;
   from->ref_cnt = from->ref_cnt + 1;
   (*mat)->data = &(from->data[offset]);
   return 0;
}


/*
 * You need to make sure that you only free `mat->data` if `mat` is not a slice and has no existing slices,
 * or if `mat` is the last existing slice of its parent matrix and its parent matrix has no other references
 * (including itself). You cannot assume that mat is not NULL.
 */
void deallocate_matrix(matrix *mat) {
     if (mat == NULL){
       return;
   }

   if ((mat->parent == NULL) & (mat->ref_cnt <= 1)){
       free(mat->data);
       free(mat);
       return;
   } else if((mat->parent == NULL) & (mat->ref_cnt > 1)){
       mat->ref_cnt -= 1;
       return;
   }else if((mat->parent != NULL) & ((mat->parent)->ref_cnt <= 1)){
       free((mat->parent)->data);
       free(mat->parent);
       free(mat);
   } else if ((mat->parent != NULL) & ((mat->parent)->ref_cnt > 1)){
       (mat->parent)->ref_cnt -= 1;
       free(mat);
       return;
   }
   return;
}


/*
 * Returns the double value of the matrix at the given row and column.
 * You may assume `row` and `col` are valid.
 */
double get(matrix *mat, int row, int col) {
   int colmat = mat->cols;
   int index = (colmat * row) + col;
   return mat->data[index];
   
}


/*
 * Sets the value at the given row and column to val. You may assume `row` and
 * `col` are valid
 */
void set(matrix *mat, int row, int col, double val) {
    int colmat = mat->cols;
    int index = (colmat * row) + col;
    mat->data[index] = val;
}


/*
 * Sets all entries in mat to val
 */
void fill_matrix(matrix *mat, double val) {
    if ((mat == NULL) | (mat->data == NULL)){
       return;
   }
   int len = (mat->cols) * (mat->rows);
    if (len < 100){
   for (unsigned int i = 0; i < len;i++){
       mat->data[i] = val;
   }
    }
   
   else
   {
     
     int i;
     __m256d tempload = _mm256_set1_pd(val);
     #pragma omp parallel shared(val,tempload) private(i)
     {
      #pragma omp for
      for (unsigned int i = 0; i < (len / 4)*4; i +=4){
          _mm256_storeu_pd( &(mat->data[i]), tempload);
  }
  
     }
  for (unsigned int t = (len/ 4) * 4; t < len; t++){
      mat->data[t] = val;
  }

   
   
   }
   
  
}


/*
 * Store the result of adding mat1 and mat2 to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 */
int add_matrix(matrix *result, matrix *mat1, matrix *mat2) {
    if ((mat1 == NULL) | (mat2 == NULL)){
        return -1;
    }
    if ((mat1->cols != mat2->cols) | (mat1->rows != mat2->rows)){
        return -1;
    }
    if ((mat1->data == NULL) | (mat2->data == NULL)){
        return -1;
    }
    int len = (mat1->rows) * (mat1->cols);
    
    if (len < 100){

    for (unsigned int i = 0; i < len; i++){
        result->data[i] = mat1->data[i] + mat2->data[i];
    }
    }

     else
  {
      __m256d temp1;
      __m256d temp2; 
      __m256d temp3;

      __m256d temp1_a;
      __m256d temp2_a; 
      __m256d temp3_a; 


      __m256d temp1_b;
      __m256d temp2_b; 
      __m256d temp3_b; 


      __m256d temp1_c;
      __m256d temp2_c; 
      __m256d temp3_c;  
      int i;

    #pragma omp parallel private(i,temp1,temp2,temp3,temp1_a,temp2_a,temp3_a,temp1_b,temp2_b,temp3_b,temp1_c,temp2_c,temp3_c)      

      {
      #pragma omp for schedule(static)
      for (unsigned int i = 0; i < (len / 16)*16; i +=16){
          temp1 = _mm256_loadu_pd (&mat1->data[i]);
          temp1_a = _mm256_loadu_pd (&mat1->data[i+4]);
          temp1_b = _mm256_loadu_pd (&mat1->data[i+8]);
          temp1_c = _mm256_loadu_pd (&mat1->data[i+12]);
          
          
          temp2 = _mm256_loadu_pd (&mat2->data[i]);
          temp2_a = _mm256_loadu_pd (&mat2->data[i+4]);
          temp2_b = _mm256_loadu_pd (&mat2->data[i+8]);
          temp2_c = _mm256_loadu_pd (&mat2->data[i+12]);
          
          
          temp3 = _mm256_add_pd (temp1, temp2);
          temp3_a = _mm256_add_pd (temp1_a, temp2_a);
          temp3_b = _mm256_add_pd (temp1_b, temp2_b);
          temp3_c = _mm256_add_pd (temp1_c, temp2_c);

    
          _mm256_storeu_pd(&(result->data[i]), temp3);
          _mm256_storeu_pd(&(result->data[i+4]), temp3_a);
          _mm256_storeu_pd(&(result->data[i+8]), temp3_b);
          _mm256_storeu_pd(&(result->data[i+12]), temp3_c);

  }
      }
  
  for (unsigned int t = (len/ 16) * 16; t < len; t++){
      result->data[t] = mat1->data[t] + mat2->data[t];
  }
  }

    return 0;

}

/*
 * Store the result of subtracting mat2 from mat1 to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 */
int sub_matrix(matrix *result, matrix *mat1, matrix *mat2) {
    if ((mat1 == NULL) | (mat2 == NULL)){
        return -1;
    }
    if ((mat1->cols != mat2->cols) | (mat1->rows != mat2->rows)){
        return -1;
    }
    if ((mat1->data == NULL) | (mat2->data == NULL)){
        return -1;
    }
    
    int len = (mat1->rows) * (mat1->cols);
    
    if (len < 100){

    for (unsigned int i = 0; i < len; i++){
        result->data[i] = mat1->data[i] - mat2->data[i];
    }
    }

     else
  {
       __m256d temp1;
      __m256d temp2; 
      __m256d temp3;

      __m256d temp1_a;
      __m256d temp2_a; 
      __m256d temp3_a; 


      __m256d temp1_b;
      __m256d temp2_b; 
      __m256d temp3_b; 


      __m256d temp1_c;
      __m256d temp2_c; 
      __m256d temp3_c;  
      int i;

        #pragma omp parallel private(i,temp1,temp2,temp3,temp1_a,temp2_a,temp3_a,temp1_b,temp2_b,temp3_b,temp1_c,temp2_c,temp3_c)
      {
      #pragma omp for
      for (unsigned int i = 0; i < (len / 16)*16; i +=16){
         temp1 = _mm256_loadu_pd (&mat1->data[i]);
          temp1_a = _mm256_loadu_pd (&mat1->data[i+4]);
          temp1_b = _mm256_loadu_pd (&mat1->data[i+8]);
          temp1_c = _mm256_loadu_pd (&mat1->data[i+12]);
          
          
          temp2 = _mm256_loadu_pd (&mat2->data[i]);
          temp2_a = _mm256_loadu_pd (&mat2->data[i+4]);
          temp2_b = _mm256_loadu_pd (&mat2->data[i+8]);
          temp2_c = _mm256_loadu_pd (&mat2->data[i+12]);
          
          
          temp3 = _mm256_sub_pd (temp1, temp2);
          temp3_a = _mm256_sub_pd (temp1_a, temp2_a);
          temp3_b = _mm256_sub_pd (temp1_b, temp2_b);
          temp3_c = _mm256_sub_pd (temp1_c, temp2_c);

    
          _mm256_storeu_pd(&(result->data[i]), temp3);
          _mm256_storeu_pd(&(result->data[i+4]), temp3_a);
          _mm256_storeu_pd(&(result->data[i+8]), temp3_b);
          _mm256_storeu_pd(&(result->data[i+12]), temp3_c);
  }
  
        }
  for (unsigned int t = (len/ 16) * 16; t < len; t++){
      result->data[t] = mat1->data[t] - mat2->data[t];
  }
  }

    return 0;

}

/*
 * Store the result of multiplying mat1 and mat2 to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 * Remember that matrix multiplication is not the same as multiplying individual elements.
 */

int mul_matrix(matrix *result, matrix *mat1, matrix *mat2) {
    if ((mat1 == NULL) | (mat2 == NULL)){
        return -1;
    }
    if ((mat1->data == NULL) | (mat2->data == NULL)){
        return -1;
    }
    int row1 = mat1->rows;
    int col1 = mat1->cols;
    int row2 = mat2->rows;
    int col2 = mat2->cols;
    
    if ((row1<1) | (row2<1) | (col1<1) | (col2<1)){
        return -1;
    }
    if (col1 != row2){
        return -1;
    }
    
    double *matrix1= mat1->data;
    
    int len = row2*col2;
    double* matrix2 = malloc(sizeof(double)*len);
    int o;
    int p;

    
    #pragma omp parallel private(o,p) shared(matrix2)
    {
    #pragma omp for
    for (unsigned int o = 0; o < col2; o++){
        for (unsigned int p = 0; p<row2;p++){
            matrix2[o*row2 + p] = mat2->data[o+p*col2];
    }
    }
    }
    
    
    double *dest = result->data;
    
    int i;
    int j;
    int k;
    __m256d temp1;
      __m256d temp2; 
      __m256d temp3;
      double * adder;
      int le;
      int po;
      double sum;
      
   
   
   #pragma omp parallel private(i,j,k,temp1,temp2,temp3,adder,le,po,sum) shared(dest,matrix1,matrix2)
   {
    #pragma omp for
    for (unsigned int i = 0; i < row1;i++){
        for(unsigned int j = 0; j < col2;j++){
            temp3 = _mm256_set1_pd(0.0);
            sum = 0;
            for(unsigned int k = 0; k <(col1/4)*4; k += 4){
                temp1 = _mm256_loadu_pd (&matrix1[i*col1 + k]);
                temp2 = _mm256_loadu_pd (&matrix2[j*col1+k]);
                temp3 = _mm256_fmadd_pd(temp1, temp2,temp3);
            
            }
            for (unsigned int le = (col1/4)*4; le < col1; le++)
            {
                sum += matrix1[i*col1 + le] * matrix2[j*col1+le];
            }
            adder = malloc(sizeof(double)*4);
            _mm256_storeu_pd(adder, temp3);
            for (unsigned int po = 0; po < 4; po++)
            {
                sum += adder[po];
            }
            dest[i*col2+j] = sum;
            free(adder);
        }
    }
   }
    free(matrix2);
    
    return 0;


}

int mul_matrixhelp(double * result, double *mat1, double *mat2, int dim) {
    
    double *matrix1= mat1;
    
    int len = dim*dim;
    double* matrix2 = malloc(sizeof(double)*len);
    int o;
    int p;

    
    #pragma omp parallel private(o,p) shared(matrix2)
    {
    #pragma omp for
    for (unsigned int o = 0; o < dim; o++){
        for (unsigned int p = 0; p<dim;p++){
            matrix2[o*dim + p] = mat2[o+p*dim];
    }
    }
    }
    
    
    double *dest = result;
    
    int i;
    int j;
    int k;
    __m256d temp1;
      __m256d temp2; 
      __m256d temp3;
      double * adder;
      int le;
      int po;
      double sum;
      
   
   
   #pragma omp parallel private(i,j,k,temp1,temp2,temp3,adder,le,po,sum) shared(dest,matrix1,matrix2)
   {
    #pragma omp for
    for (unsigned int i = 0; i < dim;i++){
        for(unsigned int j = 0; j < dim;j++){
            temp3 = _mm256_set1_pd(0.0);
            sum = 0;
            for(unsigned int k = 0; k <(dim/4)*4; k += 4){
                temp1 = _mm256_loadu_pd (&matrix1[i*dim + k]);
                temp2 = _mm256_loadu_pd (&matrix2[j*dim+k]);
                temp3 = _mm256_fmadd_pd(temp1, temp2,temp3);
            
            }
            for (unsigned int le = (dim/4)*4; le < dim; le++)
            {
                sum += matrix1[i*dim + le] * matrix2[j*dim+le];
            }
            adder = malloc(sizeof(double)*4);
            _mm256_storeu_pd(adder, temp3);
            for (unsigned int po = 0; po < 4; po++)
            {
                sum += adder[po];
            }
            dest[i*dim+j] = sum;
            free(adder);
        }
    }
   }
    free(matrix2);
    
    return 0;


}



/*
 * Store the result of raising mat to the (pow)th power to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 * Remember that pow is defined with matrix multiplication, not element-wise multiplication.
 */
int pow_matrix(matrix *result, matrix *mat, int pow) {
    if ((mat == NULL) | (mat == NULL)) {
        return -1;
    }
    if ((mat->rows) != (mat->cols)){
        return -1;
    }
    
    int row1 = mat->rows;
    int col1 = mat->cols;    
    int len = row1 * col1;
    if (pow < 0){
        return -1;
    } 
    
       int i;
        #pragma omp parallel private(i)
        {
            #pragma omp for
        for(unsigned int i = 0; i < row1;i++){
            result->data[row1*i + i] = 1;
        }
        }
    
    
    
    if (pow == 0){    
        return 0;
    }   
    

   double *tempd = malloc(sizeof(double) * (len));
    
   #pragma omp parallel
   {
   #pragma omp for
   for(unsigned int e = 0; e<len;e++){
       tempd[e] = mat->data[e];
   }
   }



    double * reshold = malloc(sizeof(double)*len);
    double * res = result->data;
    double * hold =NULL;
    double *temphold = malloc(sizeof(double)*len);


    for(unsigned int i = pow; (i > 0);i = i / 2){
        if( i % 2 == 1){
            mul_matrixhelp(reshold,res,tempd,row1);
            hold = res;
            res = reshold;
            reshold = hold;

            
        }
        mul_matrixhelp(temphold,tempd,tempd,row1);
            hold = temphold;
            temphold = tempd;
            tempd = hold;
  
    }

    result->data = res;
    free(reshold);
    free(temphold);

    

   
    
        
    
    
    return 0;


}

/*
 * Store the result of element-wise negating mat's entries to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 */
int neg_matrix(matrix *result, matrix *mat) {
    if ((mat == NULL )| (result == NULL)){
        return -1;
    }
    if (mat->data == NULL)
    {
        return -1;
    }
    if ((mat->cols != result->cols) | (mat->rows != result->rows))
    {
        return -1;
    }
    int len = (mat->rows) * (mat->cols);

    if (len < 100){

     for (unsigned int i = 0; i < len; i++){
        result->data[i] = -(mat->data[i]);
    }
    }

     else
  {
      __m256d temp1;
      __m256d temp2 =  _mm256_set1_pd(0.0);
      __m256d temp3;
      __m256d temp1a; 
      __m256d temp3a;
      __m256d temp1b; 
      __m256d temp3b;
      __m256d temp1c; 
      __m256d temp3c;     
      
      int i;
    
      #pragma omp parallel private(temp1,temp2,temp3,temp1a,temp1b,temp1c,temp3a,temp3b,temp3c,i)
      {
        #pragma omp for
      for (unsigned int i = 0; i < (len / 16)*16; i +=16){
          temp1 = _mm256_loadu_pd (&mat->data[i]);
          temp1a = _mm256_loadu_pd (&mat->data[i+4]);
          temp1b = _mm256_loadu_pd (&mat->data[i+8]);
          temp1c = _mm256_loadu_pd (&mat->data[i+12]);
          
          
          temp3 = _mm256_sub_pd (temp2, temp1);
          temp3a = _mm256_sub_pd (temp2, temp1a);
          temp3b = _mm256_sub_pd (temp2, temp1b);
          temp3c = _mm256_sub_pd (temp2, temp1c);
          
          
          
          _mm256_storeu_pd(&(result->data[i]), temp3);
          _mm256_storeu_pd(&(result->data[i+4]), temp3a);
          _mm256_storeu_pd(&(result->data[i+8]), temp3b);
          _mm256_storeu_pd(&(result->data[i+12]), temp3c);
  }
      
      
      
      }
  
  
  for (unsigned int t = (len/ 16) *16; t < len; t++){
        result->data[t] = -(mat->data[t]);
  }
  }

    return 0;




}

/*
 * Store the result of taking the absolute value element-wise to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 */
int abs_matrix(matrix *result, matrix *mat) {
    if ((mat == NULL )| (result == NULL)){
        return -1;
    }
    if (mat->data == NULL)
    {
        return -1;
    }
    if ((mat->cols != result->cols) | (mat->rows != result->rows))
    {
        return -1;
    }
    int len = (mat->rows) * (mat->cols);
    double temp;
    int i;
    __m256d temp1;
    __m256d temp1a;
    __m256d temp1b;
    __m256d temp1c;

    __m256d  temp2  =  _mm256_set1_pd(0.0);
      __m256d temp3;
      __m256d temp3a; 
      __m256d temp3b; 
      __m256d temp3c;  
      
      
      
      __m256d temp4;
      __m256d temp4a;
      __m256d temp4b;
      __m256d temp4c;


    #pragma omp parallel private(temp1,temp3,temp4,i) shared(temp2)
    {
    #pragma omp for    
    for (unsigned int i = 0; i < (len/16)*16; i = i + 16){
       
        temp1 = _mm256_loadu_pd (&(mat->data[i]));
        temp1a = _mm256_loadu_pd (&(mat->data[i+4]));
        temp1b = _mm256_loadu_pd (&(mat->data[i+8]));
        temp1c = _mm256_loadu_pd (&(mat->data[i+12]));
        
        temp3 = _mm256_sub_pd (temp2, temp1);
        temp3a = _mm256_sub_pd (temp2, temp1a);
        temp3b = _mm256_sub_pd (temp2, temp1b);
        temp3c = _mm256_sub_pd (temp2, temp1c);
        
        
        temp4 =  _mm256_max_pd (temp1, temp3);
        temp4a =  _mm256_max_pd (temp1a, temp3a);
        temp4b =  _mm256_max_pd (temp1b, temp3b);
        temp4c =  _mm256_max_pd (temp1c, temp3c);
        
        
        _mm256_storeu_pd(&(result->data[i]), temp4);
        _mm256_storeu_pd(&(result->data[i+4]), temp4a);
        _mm256_storeu_pd(&(result->data[i+8]), temp4b);
        _mm256_storeu_pd(&(result->data[i+12]), temp4c);
        
    }
    }
    double val;
    for (unsigned int t =(len/16)*16; t < len; t++)
    {
        val = mat->data[t];
        if (val<0){
            result->data[t] = -1*val;
        }else{
            result->data[t] = val;
        }
        
        
    }
    
    return 0;
}

