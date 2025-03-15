#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct node {
    char name[50];  // Allocating fixed space for name
    int age;
    int rollno;
    float os_marks;
    float dsa_marks;
    float discrete_marks;
    float lab_marks;
    float total_marks;
    struct node *link;
};

struct node *head = NULL;
// Function to write data to a file
void writeToFile() {
    FILE *file = fopen("C:/Users/hp/CLionProjects/linklist/cmake-build-debug/students.txt", "w");  // Open file in write mode
    if (file == NULL) {
        printf("Error opening file!\n");
        return;
    }

    struct node *temp = head;
    while (temp != NULL) {
        fprintf(file, "%s %d %d %.2f %.2f %.2f %.2f\n",
                temp->name, temp->age, temp->rollno,
                temp->os_marks, temp->dsa_marks, temp->discrete_marks, temp->lab_marks);
        temp = temp->link;
    }

    fclose(file);
    printf("Data successfully written to students.txt\n");
}
void remove_outliers() {
    struct node *p=head,*q;
    char name[100];
    printf("Enter the name of the student");
    scanf("%s",name);
    while (p!=NULL) {
        if (p->name==name) {
            q->link=p->link;
            free(p);
        }
        p=p->link;
        q=p;
    }
}
void insert_at_beginning() {
    struct node *newnode,*temp=head;
    newnode = (struct node *)malloc(sizeof(struct node));
    printf("Enter Name, Age, Roll No, OS Marks, DSA Marks, Discrete Marks, Lab Marks:\n");
    scanf("%s %d %d %f %f %f %f", newnode->name, &newnode->age, &newnode->rollno,
          &newnode->os_marks, &newnode->dsa_marks, &newnode->discrete_marks, &newnode->lab_marks);

    newnode->link = NULL;

 newnode->link = head;
    head = newnode;
    printf("Insert New Node Successfully\n");
}

void insert_at_middle() {
    struct node *newnode,*temp=head;
    int i=1,loc;
    newnode = (struct node *)malloc(sizeof(struct node));
    printf("Enter Name, Age, Roll No, OS Marks, DSA Marks, Discrete Marks, Lab Marks:\n");
    scanf("%s %d %d %f %f %f %f", newnode->name, &newnode->age, &newnode->rollno,
          &newnode->os_marks, &newnode->dsa_marks, &newnode->discrete_marks, &newnode->lab_marks);
    newnode->link = NULL;
    printf("Enter the location");
    scanf("%d",&loc);
    while (i<loc-1) {
        temp=temp->link;
        i++;
    }
    newnode->link = temp->link;
    temp->link = newnode;
}
void insertNode() {
    struct node *newnode = (struct node *)malloc(sizeof(struct node));
    if (newnode == NULL) {
        printf("Memory allocation failed\n");
        return;
    }

    printf("Enter Name, Age, Roll No, OS Marks, DSA Marks, Discrete Marks, Lab Marks:\n");
    scanf("%s %d %d %f %f %f %f", newnode->name, &newnode->age, &newnode->rollno,
          &newnode->os_marks, &newnode->dsa_marks, &newnode->discrete_marks, &newnode->lab_marks);

    newnode->link = NULL;

    if (head == NULL) {
        head = newnode;
    } else {
        struct node *temp = head;
        while (temp->link != NULL) {
            temp = temp->link;
        }
        temp->link = newnode;
    }
    writeToFile();
}

void displayNode() {
    struct node *temp = head;
    if (temp == NULL) {
        printf("No records found.\n");
        return;
    }

    printf("\n%-10s %-5s %-10s %-10s %-10s %-10s %-10s %-10s\n",
           "Name", "Age", "Roll No", "OS Marks", "DSA Marks", "Discrete Marks", "Lab Marks","Total Marks");
    printf("---------------------------------------------------------------\n");

    while (temp != NULL) {
        printf("%-10s %-5d %-10d %-10.2f %-10.2f %-10.2f %-10.2f %10.2f\n",
               temp->name, temp->age, temp->rollno, temp->os_marks,
               temp->dsa_marks, temp->discrete_marks, temp->lab_marks,temp->total_marks);
        temp = temp->link;
    }
}

int length() {
    struct node *temp = head;
    int len = 0;
    while (temp) {
        len++;
        temp = temp->link;
    }
    return len;
}
void total_marks() {
    struct node *temp = head;
    while (temp != NULL) {
        temp->total_marks=temp->dsa_marks+temp->discrete_marks+temp->lab_marks+temp->os_marks;
        temp = temp->link;
    }
}
void deleteNode() {
    if (head == NULL) {
        printf("List is empty, nothing to delete.\n");
        return;
    }

    int loc;
    printf("Enter the location to delete (1-based index): ");
    scanf("%d", &loc);

    if (loc < 1 || loc > length()) {
        printf("Invalid location\n");
        return;
    }

    struct node *temp = head, *prev = NULL;

    if (loc == 1) {  // Deleting the first node
        head = head->link;
        free(temp);
        printf("Node deleted at position %d\n", loc);
        return;
    }

    for (int i = 1; temp != NULL && i < loc; i++) {
        prev = temp;
        temp = temp->link;
    }

    if (temp == NULL) return; // This should never happen due to previous check

    prev->link = temp->link;
    free(temp);
    printf("Node deleted at position %d\n", loc);
}
void Sorted_list() {
    struct node *p, *q;
    char temp_name[50];
    int temp_age, temp_rollno;
    float temp_os, temp_dsa, temp_discrete, temp_lab;

    if (head == NULL || head->link == NULL) return; // If list is empty or has one node, no need to sort.

    for (p = head; p->link != NULL; p = p->link) {
        for (q = head; q->link != NULL; q = q->link) {
            if (strcmp(q->name, q->link->name) > 0) { // Compare names
                // Swap names
                strcpy(temp_name, q->name);
                strcpy(q->name, q->link->name);
                strcpy(q->link->name, temp_name);

                // Swap other data
                temp_age = q->age;
                q->age = q->link->age;
                q->link->age = temp_age;

                temp_rollno = q->rollno;
                q->rollno = q->link->rollno;
                q->link->rollno = temp_rollno;

                temp_os = q->os_marks;
                q->os_marks = q->link->os_marks;
                q->link->os_marks = temp_os;

                temp_dsa = q->dsa_marks;
                q->dsa_marks = q->link->dsa_marks;
                q->link->dsa_marks = temp_dsa;

                temp_discrete = q->discrete_marks;
                q->discrete_marks = q->link->discrete_marks;
                q->link->discrete_marks = temp_discrete;

                temp_lab = q->lab_marks;
                q->lab_marks = q->link->lab_marks;
                q->link->lab_marks = temp_lab;
            }
        }
    }
}
int main() {
    int ch;
    while (1) {
        printf("\n1. Insert\n2. Display\n3. Delete\n4.Insertion_At_Beginning\n5. Insertion_At_Middle\n6. Calculate_Total_Marks\n7. Sorted List\n8. Exit\n");
        printf("Enter choice: ");
        scanf("%d", &ch);
        switch (ch) {
            case 1:
                insertNode();
                break;
            case 2:
                displayNode();
                break;
            case 3:
                deleteNode();
                break;
            case 4:
                insert_at_beginning();
            break;
            case 5:
                insert_at_middle();
            break;
            case 6:
                total_marks();
            break;
            case 7:
                Sorted_list();
            break;
            case 8:
                exit(0);
            default:
                printf("Invalid choice. Please try again.\n");
        }
    }
    return 0;
}
