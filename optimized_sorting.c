#include<stdio.h>
#include<stdlib.h>
//Bubble Sorting Technique
int* bubblesort(int arr[],int n)
{
    int* sortedArr = (int*)malloc(n * sizeof(int)); // Allocate memory for sorted array
    if (sortedArr == NULL) {
        printf("Memory allocation failed!\n");
        return NULL;
    }
    for (int i = 0; i < n; i++)
      sortedArr[i] = arr[i];

    for (int i = 1; i < n-1; i++) {
      for (int j = 0; j < n-i-1; j++) {
        if (sortedArr[j] > sortedArr[j+1]) {
          int temp = sortedArr[j];
          sortedArr[j] = sortedArr[j+1];
          sortedArr[j+1] = temp;
        }
      }
    }
return sortedArr;
}
//Quick Sort Algorithm
void swap(int *a, int *b) {
  int temp = *a;
  *a = *b;
  *b = temp;
}

void printArr(int arr[],int n)
{
  for (int i = 0; i < n; i++) {
    printf("%d ", arr[i]);
  }
  printf("\n");
}
int partition(int arr[], int low, int high) {
  int pivot = arr[high]; // Choose last element as pivot
  int i = (low - 1); // Index for smaller element

  for (int j = low; j < high; j++) {
    if (arr[j] < pivot) { // If current element is smaller than pivot
      i++;
      swap(&arr[i], &arr[j]);
    }
  }
  swap(&arr[i + 1], &arr[high]); // Swap pivot to correct position
  return (i + 1);
}
int* quickSort(int arr[], int low, int high) {
  if (low < high) {
    int pi = partition(arr, low, high); // Partition index

    quickSort(arr, low, pi - 1); // Sort left part
    quickSort(arr, pi + 1, high); // Sort right part
  }
  return arr;
}

int main()
    {
 int arr[]={32,22,54,76,65,54,34,98,89,67,56,45,23};
 int len=sizeof(arr)/sizeof(arr[0]);
  int ch;
  while (1) {
    printf("1. Bubble Sort\n");
    printf("2. Quick Sort\n");
    printf("Exit");
    scanf("%d,&ch");
    switch (ch) {
      case 1:

        int* sortedArr = bubblesort(arr,len);
      if(sortedArr!=NULL)
      {
        printf("The sorted array is:\n");
        printArr(sortedArr,len);
        free(sortedArr);
      }break;
      case 2:
        int* QuickSort = QuickSort(arr,0,len-1);
      if(QuickSort!=NULL) {
        printf("The sorted array is:\n");
        printArr(QuickSort,len);
        free(QuickSort);
      }break;
    }
  }
    return 0;
  }