#include<stdio.h>
#include<stdlib.h>
int strlen(){
    char str[100];
    printf("Enter String:\n");
    scanf("%[^\n]", str);
    int i=0;
    while(str[i]!='\0'){
        i++;
    }
    return i;
}
int main(){
    printf("%d", strlen());
}