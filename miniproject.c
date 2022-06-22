// SENARIO BASED MINI PROJECT
//c program to encrypt, decrypt and masking of a password
#include <stdio.h>
#include <string.h>
/* EXPAINATION AND SENARIO OF THE CODE- every company have a different set of password for a particular user, 
   and most of the time it can be quite easy for a 3rd party to retrive data of the if he can figure out the password by cracking the code.
   so a company adds its own characters into the password known as THE KEY, to make sure that it becomes harder for a third party to crack the code.
   here in this code we are recieving a sort of password from the user that we will encrypt by adding and converting the password into some
   other code using the key, and then later on when to decrypt the password we will subsract the same key to show the origional password.
   the password is converted in another form as every single characters, special characters have a ASCII value assigned to it so,
   using the ascii value assinged to the characters in the password and key, c will simply add and subract the ascii numbers and then give
   us the output in the form of characters.
 */
void encrypt(char password[],int key)
{
    unsigned int i;
    for(i=0;i<strlen(password);++i)
    {
        password[i] = password[i] - key;// here in the encryption we are reciving a string of characters whose no of characters are counted using strlen and then the new encrypted password is created by subtracting the key we will define from the string user entered
    }
}
void decrypt(char password[],int key)
{
    unsigned int i;
    for(i=0;i<strlen(password);++i)
    {
        password[i] = password[i] + key;// here in decryption we are reciving a the new password after encryption, here also we count the no of characters and since we subtracted the key from the origional string the user gave here we will add the key to the encrypted password and we will get the origional password back
    }
}
int main()
{
    char password[50] ;
    printf("Enter the password: \n ");
    scanf("%s", password);
    printf("Password     = %s\n",password);
    encrypt(password,0xAED); //here 0xAED is the key that we have used for encryption, we can take any form of key as long as it is under the conditions we entered
    printf("Encrypted value = %s\n",password);
    decrypt(password,0xAED);
    printf("Decrypted value = %s\n",password);
    return 0;
}