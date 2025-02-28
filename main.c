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
    struct node *link;
};

struct node *head = NULL;
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
}

void displayNode() {
    struct node *temp = head;
    if (temp == NULL) {
        printf("No records found.\n");
        return;
    }

    printf("\n%-10s %-5s %-10s %-10s %-10s %-10s %-10s\n",
           "Name", "Age", "Roll No", "OS Marks", "DSA Marks", "Discrete Marks", "Lab Marks");
    printf("---------------------------------------------------------------\n");

    while (temp != NULL) {
        printf("%-10s %-5d %-10d %-10.2f %-10.2f %-10.2f %-10.2f\n",
               temp->name, temp->age, temp->rollno, temp->os_marks,
               temp->dsa_marks, temp->discrete_marks, temp->lab_marks);
        temp = temp->link;
    }
}

int length() {
    struct node *temp = head;
    int len = 0;
    while (temp != NULL) {
        len++;
        temp = temp->link;
    }
    return len;
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

int main() {
    int ch;
    while (1) {
        printf("\n1. Insert\n2. Display\n3. Delete\n4.Insertion_At_Beginning\n5. Insertion_At_Middle\n6. Exit\n");
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
                exit(0);
            default:
                printf("Invalid choice. Please try again.\n");
        }
    }
    return 0;
}
