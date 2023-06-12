#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <fcntl.h>
#include <unistd.h>

#include <sys/stat.h>
#include <utime.h>

#define MAX_LINE_LENGTH 1024
#define BUFFER_SIZE 100
#define NUM_BUFFERS 100
#define NUM_FILES 100
#define NUM_FDESCRIPTORS 100

int file_descriptor[NUM_FDESCRIPTORS];
char* buffer[NUM_BUFFERS];
char* file_names[NUM_FILES];

void execute_command(char* command) {
    printf("Executing command: %s\n", command);
    int operation, transaction, buffer_id, name_id, name2_id, offset, size, fd;
    sscanf(command, "%d%d%d%d%d%d%d", &operation, &transaction, &buffer_id, &name_id, &name2_id, &offset, &size);

    switch (operation) {
        case 0:
            printf("init %d\n", transaction);

            fd = open(file_names[name_id], O_RDWR | O_CREAT, 0666);
            ftruncate(fd, offset);
            close(fd);

            break;
        case 1:
            printf("open %d\n", transaction);

            file_descriptor[transaction] = open(file_names[name_id], O_RDWR | O_CREAT, 0666);

            break;
        
        case 2:
            printf("close %d\n", transaction);

            close(file_descriptor[transaction]);

            break;

        case 3:
            printf("write %d\n", transaction);

            write(file_descriptor[transaction], buffer[buffer_id], size);

            break;

        case 4:
            printf("read %d\n", transaction);

            read(file_descriptor[transaction], buffer[buffer_id], size);

            break;

        case 5:
            printf("seek %d\n", transaction);

            lseek(file_descriptor[transaction], offset, SEEK_SET);

            break;
        
        case 6:
            printf("truncate %d\n", transaction);

            ftruncate(file_descriptor[transaction], offset);

            break;
        
        case 7:
            printf("sync %d\n", transaction);

            fsync(file_descriptor[transaction]);

            break;

        case 8:
            printf("delete %d\n", transaction);

            remove(file_names[name_id]);

            break;
        

        case 9:
            printf("rename %d\n", transaction);

            rename(file_names[name_id], file_names[name2_id]);

            break;

        case 10:
            printf("link %d\n", transaction);

            symlink(file_names[name_id], file_names[name2_id]);

            break;

        case 11:
            printf("unlink %d\n", transaction);

            unlink(file_names[name_id]);

            break;

        case 12:
            printf("mkdir %d\n", transaction);

            // mkdir("dir", 0777);

            break;

        case 13:
            printf("rmdir %d\n", transaction);

            rmdir("dir");

            break;

        case 14:
            printf("chdir %d\n", transaction);

            chdir("dir");

            break;

        case 15:
            printf("chmod %d\n", transaction);

            chmod(file_names[name_id], 0777);

            break;

        case 16:
            printf("chown %d\n", transaction);

            chown(file_names[name_id], 1000, 1000);

            break;
        
        case 17:
            printf("utime %d\n", transaction);

            struct utimbuf time;
            time.actime = 1000;
            time.modtime = 1000;
            utime(file_names[name_id], &time);

            break;
        
        case 18:
            printf("stat %d\n", transaction);

            struct stat buf;
            stat(file_names[name_id], &buf);

            break;

        case 19:
            printf("lstat %d\n", transaction);

            struct stat buf2;
            lstat(file_names[name_id], &buf2);

            break;

        case 20:
            printf("access %d\n", transaction);

            access(file_names[name_id], F_OK);

            break;
    }
}

void initialize_buffers(){
  for(int transaction = 0; transaction < NUM_FDESCRIPTORS; transaction++){
    file_descriptor[transaction] = transaction;
  }

  for(int buffer_id = 0; buffer_id < NUM_BUFFERS; buffer_id++){
    buffer[buffer_id] = (char*)malloc(BUFFER_SIZE);
    for(int offset = 0; offset < BUFFER_SIZE; offset++){
      buffer[buffer_id][offset] = rand();
    }
  }

  for( int n = 0; n < NUM_FILES; n++ ){
    file_names[n] = (char*) malloc(10);
    sprintf(file_names[n], "file_%d", n);
  }

}

int main(int argc, char* argv[]) {

    srand(1);
    initialize_buffers();

    if (argc != 2) {
        printf("Usage: %s input_file\n", argv[0]);
        return 1;
    }

    FILE* input_file = fopen(argv[1], "r");
    if (input_file == NULL) {
        printf("Error: could not open input file %s\n", argv[1]);
        return 1;
    }

    char line[MAX_LINE_LENGTH];

    time_t start_time = time(0);

    while (fgets(line, MAX_LINE_LENGTH, input_file) != NULL) {
        char* timestamp_str = strtok(line, " ");
        char* command = strtok(NULL, "\n");
        if (timestamp_str == NULL || command == NULL) {
            printf("Error: invalid input line\n");
            continue;
        }
        long long timestamp_ms = strtoll(timestamp_str, NULL, 10);

        time_t current_time = time(0);
        long elapsed_time = difftime(current_time, start_time) * 1000;
        long ms_to_wait = timestamp_ms - elapsed_time;

        if (ms_to_wait > 0) {
            printf("Waiting for %ld milliseconds...\n", ms_to_wait);
            usleep((unsigned int)(ms_to_wait * 1000));
        }
        execute_command(command);
    }

    fclose(input_file);

    int filebuffers = open("buffers", O_RDWR | O_CREAT, 0666);
    for(int buffer_id = 0; buffer_id < NUM_BUFFERS; buffer_id++){
      write(filebuffers, buffer[buffer_id], BUFFER_SIZE);
    }
    close(filebuffers);

    return 0;
}
