/**********************************************************
"Hello World"-type program to test different srun layouts.

Written by Tom Papatheodore
**********************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <iomanip>
#include <string.h>
#include <mpi.h>
#include <sched.h>
#include <hip/hip_runtime.h>
#include <omp.h>

// #define DEBUG
#define BARDPEAK

#define EXIT_SUCCESS         0
#define EXIT_WRONG_ARGUMENT  1
#define EXIT_HIP_ERROR       2
#define EXIT_SEND_ERROR      3
#define EXIT_RECV_ERROR      4
#define EXIT_INTERNAL_ERROR  5

#if defined( BARDPEAK )
//
// Settings for the Cray EX Bardpeak GPU nodes of LUMI and similar machines
//
#define MAXGPUS                8
#define SINGLE_PCIBUSID_STRLEN 13  // Full bus IDs of the form: 0000:c1:00.0
#define LIST_PCIBUSID_STRLEN   MAXGPUS * (13 + 1) // 13: length of the items in the map below.
                                                  // 1: , and trailing \0 after last item
const char * const busid_map[MAXGPUS] = {
		"0000:c1:00.0",
		"0000:c6:00.0",
		"0000:c9:00.0",
		"0000:ce:00.0",
		"0000:d1:00.0",
		"0000:d6:00.0",
		"0000:d9:00.0",
		"0000:de:00.0"
    };
const char * const busid_map_values_short[MAXGPUS] = {
		"c1",
		"c6",
		"c9",
		"ce",
		"d1",
		"d6",
		"d9",
		"dc"
    };
const char * const busid_map_values_long[MAXGPUS] = {
	    "c1(GCD0/CCD6)",
		"c6(GCD1/CCD7)",
		"c9(GCD2/CCD2)",
		"cc(GCD3/CCD3)",
		"d1(GCD4/CCD0)",
		"d6(GCD5/CCD1)",
		"d9(GCD6/CCD4)",
		"dc(GCD7/CCD5)"
    };

#define L3DOMAINS 8

// MPI 000 - OMP 000 - HWT 001 (CCD0) - Node nid005952 - RT_GPU_ID 0,1,2,3,4,5,6,7 - GPU_ID 0,1,2,3,4,5,6,7 - Bus_ID c1(GCD0/CCD6),c6(GCD1/CCD7),c9(GCD2/CCD2),cc(GCD3/CCD3),d1(GCD4/CCD0),d6(GCD5/CCD1),d9(GCD6/CCD4),dc(GCD7/CCD5)
#define OUTPUT_LINE_LENGTH 256

unsigned int HWT_to_L3domain( unsigned int HWT ) {

	unsigned int HWT2 = HWT;

	HWT &= (unsigned int) 63; // Corresponds to computing the physical core number.
    HWT >>= 3;                // Divide by 8 to get the GCD / L3 domain

	return HWT;

}

#endif
// #endif settings for Cray EX Bardpeak nodes

typedef struct {

	char *p_IObuf;                   // Will point to the first character of the IO buffer for thread 0.

    int  mpi_myrank;                 // MPI rank of the current process
    int  openmp_numthreads;          // Number of threads in the current MPI process

    char IObuf[OUTPUT_LINE_LENGTH]; // IO string buffer for the first thread.
} t_rankData;

// Macro for checking errors in HIP API calls
#define hipErrorCheck(call)                                                                     \
do {                                                                                            \
    hipError_t hipErr = call;                                                                   \
    if( hipErr != hipSuccess ){                                                                 \
        printf( "HIP Error - %s:%d: '%s'\n", __FILE__, __LINE__, hipGetErrorString( hipErr ) ); \
        MPI_Abort( MPI_COMM_WORLD, EXIT_HIP_ERROR );                                                                 \
    }                                                                                           \
} while(0)



//******************************************************************************
//
// print_help()
//
// Prints help information.
//

void print_help( const char *exe_name ) {

	fprintf( stderr,
		"\n"
        "%s\n"
		"\n",
        exe_name
	);
	fprintf( stderr,
		"Flags accepted:\n"
		"\n"
		"  -h, --help Show help information and exit\n"
		"  -l, --fl   Shows a bit more information: CCD with the thread number\n"
		"             and GCD and optimal CCD with the PCIe bus ID\n"
		"  -u         Unsorted printing, may work around some bugs\n"
		"\n"
	);

	fprintf( stderr,
		"Meaning of the output:\n"
		"\n"
		"  MPI:       Rank of the MPI process\n"
		"  OMP:       OpenMP thread number\n"
		"  HWT:       Hardware thread\n"
		"  RT_GPU_ID: HIP runtime GPU ID, a local ID, a series starting from 0 for\n"
		"             each process.\n"
		"  GPU_ID:    Value of ROCR_VISIBLE_DEVICES. This could refer to the global\n"
	    "             GPU IDs, but a scheduler can actually remap the GPU numbering\n"
		"             to (in case of Slurm) IDs starting from 0 for each task.\n"
		"  Bus_ID:    PCIe bus ID and the only truly reliable way to identify a\n"
		"             physical GPU.\n"
		"\n"
	);

#if defined( BARDPEAK )
	fprintf( stderr,
		"Hardware mapping:\n"
		"\n"
		"  CPU die 0 providing HWT 000-007 and 064-071 to GPU die 4 with Bus_ID d1\n"
		"  CPU die 1 providing HWT 008-015 and 072-079 to GPU die 5 with Bus_ID d6\n"
		"  CPU die 2 providing HWT 016-023 and 080-087 to GPU die 2 with Bus_ID c9\n"
		"  CPU die 3 providing HWT 024-031 and 088-095 to GPU die 3 with Bus_ID ce\n"
		"  CPU die 4 providing HWT 032-039 and 096-103 to GPU die 6 with Bus_ID d9\n"
		"  CPU die 5 providing HWT 040-047 and 104-111 to GPU die 7 with Bus_ID de\n"
		"  CPU die 6 providing HWT 048-055 and 112-119 to GPU die 0 with Bus_ID c1\n"
		"  CPU die 7 providing HWT 056-063 and 120-127 to GPU die 1 with Bus_ID c6\n"
		"\n"
		"  GPU die 0 with Bus_ID c1 to CPU die 6 providing HWT 048-055 and 112-119\n"
		"  GPU die 1 with Bus_ID c6 to CPU die 7 providing HWT 056-063 and 120-127\n"
		"  GPU die 2 with Bus_ID c9 to CPU die 2 providing HWT 016-023 and 080-087\n"
		"  GPU die 3 with Bus_ID ce to CPU die 3 providing HWT 024-031 and 088-095\n"
		"  GPU die 4 with Bus_ID d1 to CPU die 0 providing HWT 000-007 and 064-071\n"
		"  GPU die 5 with Bus_ID d6 to CPU die 1 providing HWT 008-015 and 072-079\n"
		"  GPU die 6 with Bus_ID d9 to CPU die 4 providing HWT 032-039 and 096-103\n"
		"  GPU die 7 with Bus_ID de to CPU die 5 providing HWT 040-047 and 104-111\n"
        "\n"
	);
#endif

	fprintf( stderr,
			"Restrictions:\n"
			"\n"
			"  - This tool currently only works for Cray EX bardpeak nodes (Trento + 4 * MI250X).\n"
			"\n"
		);

	fflush( stderr );

}



//******************************************************************************
//
// get_args( argc, argv, int argc, char **argv, int mpi_myrank,
//		     unsigned int *show_optimap, unsigned int *unsorted_print )
//
// Gets the input arguments.
//
// Arguments:
//  * argc: Argument count from the main function
//  * argv: Argument values from the main function
//  * mpi_myrank: MPI rank to ensure that help is printed only once.
//    Since in a heterogeneous program some program arguments may only
//    be given for the second instance, we cannot avoid that other error
//    messages will be printed once for each MPI rank in the job component.
//  * show_optimap: On return nonzero if -l is specified, zero otherwise.
//

void get_args( int argc, char **argv, int mpi_myrank,
		       unsigned int *show_optimap, unsigned int *unsorted_print ) {

	char *exe_name;

	// Make sure we always return initialised variables, whatever happens.
	*show_optimap = 0;
	*unsorted_print = 0;

	// Remove the program name
	exe_name = basename( *argv++ );
	argc--;

	while ( argc-- ) {

		if ( ( strcmp( *argv, "-h") == 0 ) || ( strcmp( *argv, "--help") == 0 ) ) {
			if ( mpi_myrank == 0 ) print_help( exe_name );
			exit( EXIT_SUCCESS );
		} else if ( ( strcmp( *argv, "-l") == 0 ) || ( strcmp( *argv, "--fl") == 0 ) ) {
			*show_optimap = 1;
		} else if ( strcmp( *argv, "-u") == 0 ) {
			*unsorted_print = 1;
		} else {
			fprintf( stderr, "%s: Illegal flag found: %s\n", exe_name, *argv);
			MPI_Abort( MPI_COMM_WORLD, EXIT_WRONG_ARGUMENT );
		}

		argv++;

	}

	return;

}


//******************************************************************************
//
// int get_GCD( const char *busid, const char * const busid_map )
//
// Arguments:
//  * busid (input): The bus ID of the GCD
//  * busid_map (input): Mapping of number of the GCD (index) to the bus ID.
//
// Return value: The GCD number of -1
//

int get_GCD( const char *busid, const char * const busid_map[] ) {

    int GCD = 0;

    while ( ( GCD < MAXGPUS ) && ( strcmp( busid, busid_map[GCD] ) != 0 ) ) GCD++;

    return ( GCD < MAXGPUS ? GCD : -1);

} // end function map_gpu




//******************************************************************************
//
// Main program
//

int main(int argc, char *argv[]){

	unsigned int show_optimap;
	unsigned int unsorted_print;

	// Initialize MPI

	MPI_Init(&argc, &argv);

	int size;
	MPI_Comm_size( MPI_COMM_WORLD, &size );

	int rank;
	MPI_Comm_rank( MPI_COMM_WORLD, &rank );

	// Initialize OpenMP threading

	int num_threads;
	int max_num_threads;
    #pragma omp parallel shared( num_threads )   // The shared clause is not strictly needed as that variable will be shared by default.
    { if ( omp_get_thread_num() == 0 ) num_threads = omp_get_num_threads(); } // Must be in a parallel session to get the proper number.
    MPI_Allreduce( &num_threads, &max_num_threads, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD );

	//
	// Interpret the command line arguments
    //

    get_args( argc, argv, rank, &show_optimap, &unsorted_print );

	//
	// Data gathering
	//

	// - Name of the node (through MPI function rather than the system library gethostname).
	char name[MPI_MAX_PROCESSOR_NAME];
	int resultlength;
	MPI_Get_processor_name( name, &resultlength );

	// - If ROCR_VISIBLE_DEVICES is set, capture visible GPUs
    const char* gpu_id_list; 
    const char* rocr_visible_devices = getenv( "ROCR_VISIBLE_DEVICES" );
    if ( rocr_visible_devices == NULL ) {
        gpu_id_list = "N/A";
    } else {
        gpu_id_list = rocr_visible_devices;
    }

	// - Find how many GPUs HIP runtime says are available
	int num_devices = 0;
    if ( hipGetDeviceCount( &num_devices ) != hipSuccess ) num_devices = 0;                                                               \

    // - Get data on each of the GPUS available to the MPI rank
    char busid_array[MAXGPUS][SINGLE_PCIBUSID_STRLEN];

	for( int i=0; i<num_devices; i++ ){ // Loop over the GPUs available to each MPI rank

		hipErrorCheck( hipSetDevice(i) );

		// Get the PCIBusId for each GPU and use it to query for UUID
		hipErrorCheck( hipDeviceGetPCIBusId( busid_array[i], (int) SINGLE_PCIBUSID_STRLEN, i ) );

#ifdef DEBUG
		{
			char temp_busid[65];
			hipErrorCheck( hipDeviceGetPCIBusId( temp_busid, (int) 65, i ) );
			printf( "GPU %d: found Bus ID %s, with longer variable: %s.\n", i, busid_array[i], temp_busid );
		}
#endif

	} // end for


	//
	// Output the results
	//

	int hwthread;
	int thread_id = 0;
	unsigned int CCD;

	t_rankData *my_rankData;   // Rank data for this MPI process
	int my_rankData_size;

	//
	// Create a buffer to prepare the I/O (to have as much joint code for sorted and unsorted)
	//

	my_rankData_size = sizeof( t_rankData ) + (num_threads - 1) * OUTPUT_LINE_LENGTH * sizeof( char );
    my_rankData = (t_rankData *) malloc( my_rankData_size );
    if ( my_rankData == NULL ) { fprintf( stderr, "ERROR: Memory allocation for my_rankData failed.\n" ); return 1; };
    my_rankData->p_IObuf = (char *) &(my_rankData->IObuf);

    my_rankData->mpi_myrank = rank;
    my_rankData->openmp_numthreads = num_threads;

	if ( num_devices == 0 ) {
		#pragma omp parallel default(shared) private(hwthread, thread_id, CCD)
		{
			thread_id = omp_get_thread_num();
			hwthread = sched_getcpu();
			CCD = HWT_to_L3domain( (unsigned int) hwthread );

			my_rankData->p_IObuf[(thread_id+1) * OUTPUT_LINE_LENGTH-1] = '\0';

			if (show_optimap )
				sprintf( my_rankData->p_IObuf + thread_id * OUTPUT_LINE_LENGTH,
						 "MPI %03d - OMP %03d - HWT %03d (CCD%d) - Node %s - GPU N/A",
						 rank, thread_id, hwthread, CCD, name);
			else
				sprintf( my_rankData->p_IObuf + thread_id * OUTPUT_LINE_LENGTH,
						 "MPI %03d - OMP %03d - HWT %03d - Node %s - GPU N/A",
						 rank, thread_id, hwthread, name);

			if ( my_rankData->p_IObuf[(thread_id+1) * OUTPUT_LINE_LENGTH-1] != '\0' ) {
				fprintf( stderr, "Internal error: Increase the size of OUTPUT_LINE_LENGTH. File %s line %d.\n",
						 __FILE__, __LINE__ );
				MPI_Abort( MPI_COMM_WORLD, EXIT_INTERNAL_ERROR );
			}
		}
	} else {

		std::string busid_list = "";
		std::string rt_gpu_id_list = "";

		// Loop over the GPUs available to each MPI rank
		for(int i=0; i<num_devices; i++) {

			// Find the physical number of the GPU from the bus ID.
			int GCD = get_GCD( busid_array[i], busid_map );
			if ( GCD < 0 ) {
				fprintf( stderr, "Unrecognized bus ID '%s' - %s:%d\n", busid_array[i], __FILE__, __LINE__ );
				MPI_Abort( MPI_COMM_WORLD, EXIT_INTERNAL_ERROR );
			} // end if ( GCD < 0 )

			// Concatenate per-MPIrank GPU info into strings for print
			if (i > 0) rt_gpu_id_list.append( "," );
			rt_gpu_id_list.append(std::to_string(i));

			//std::string temp_busid( busid_array[i] );

			if (i > 0) busid_list.append(",");
			if ( show_optimap ) {
				std::string temp_busid( busid_map_values_long[GCD] );
				busid_list.append( temp_busid );
			} else {
				std::string temp_busid( busid_map_values_short[GCD] );
				busid_list.append( temp_busid );
			} // end else-part if ( show_optimap )

		} // end for( int i=0, ... )

		#pragma omp parallel default(shared) private(hwthread, thread_id)
		{
			#pragma omp critical
			{
			thread_id = omp_get_thread_num();
			hwthread = sched_getcpu();
			CCD = HWT_to_L3domain( (unsigned int) hwthread );

			if (show_optimap )
				sprintf( my_rankData->p_IObuf + thread_id * OUTPUT_LINE_LENGTH,
						 "MPI %03d - OMP %03d - HWT %03d (CCD%d) - Node %s - RT_GPU_ID %s - GPU_ID %s - Bus_ID %s",
						 rank, thread_id, hwthread, CCD, name, rt_gpu_id_list.c_str(), gpu_id_list, busid_list.c_str() );
			else
				sprintf( my_rankData->p_IObuf + thread_id * OUTPUT_LINE_LENGTH,
						 "MPI %03d - OMP %03d - HWT %03d - Node %s - RT_GPU_ID %s - GPU_ID %s - Bus_ID %s",
						 rank, thread_id, hwthread, name, rt_gpu_id_list.c_str(), gpu_id_list, busid_list.c_str() );
		   } // end #pragma omp critical
		} // end #pragma omp parallel
	} // end if (num_devices == 0 )

	//
	// Now print to stdout, either from each MPI process (unsorted) or
	// by collecting all output in rank 0.
	//
	if ( unsorted_print ) {

		for ( int i=0; i<num_threads; i++)
			printf( "%s\n", my_rankData->p_IObuf + i*OUTPUT_LINE_LENGTH );

	} else { // else-part of if ( unsorted_print )

		int error;

		const int mssgID = 1;
	    MPI_Request request;
	    MPI_Status status;

        // Send the data

        error = MPI_Isend( my_rankData, my_rankData_size, MPI_BYTE, 0, mssgID, MPI_COMM_WORLD, &request );
        if ( error ) {
        	fprintf( stderr, "Error sending data from MPI rank %d. Error code %d.\n", rank, error);
        	MPI_Abort( MPI_COMM_WORLD, EXIT_SEND_ERROR );
        }
#ifdef DEBUG
        printf( "Rank %d: Sending %d bytes of data.\n", rank, my_rankData_size );
#endif

        // MPI_Barrier( MPI_COMM_WORLD ); // Looks to be safer to ensure that all Isends are out, some problems that
                                       // we have observed may be due to that.

        // On rank 0: Receive the data and print.

        if ( rank == 0 ) {

    		t_rankData *buf_rankData;  // Buffer to receive data from another process.
    		int buf_rankData_size;

    		// Create the buffer.

    		buf_rankData_size = sizeof( t_rankData ) + (max_num_threads - 1) * OUTPUT_LINE_LENGTH * sizeof( char );
            buf_rankData = (t_rankData *) malloc( buf_rankData_size );
            if ( buf_rankData == NULL ) { fprintf( stderr, "ERROR: Memory allocation for buf_rankData failed.\n" ); return 1; };
            buf_rankData->p_IObuf = (char *) &(buf_rankData->IObuf);

        	for ( int c1 = 0; c1 < size; c1++ ) { // Loop over all ranks.

        		error = MPI_Recv( (void *) buf_rankData, buf_rankData_size, MPI_BYTE, c1, mssgID, MPI_COMM_WORLD, &status );
        		if ( error ) {
        			fprintf( stderr, "Error receiving data from MPI rank %d. Error code %d.\n", c1, error);
        			MPI_Abort( MPI_COMM_WORLD, EXIT_RECV_ERROR );
        		}
                buf_rankData->p_IObuf = (char *) &(buf_rankData->IObuf); // Need to redo this as it is overwritten by the communication.
#ifdef DEBUG
                {
                	int count;
                	MPI_Get_count( &status, MPI_BYTE, &count );
                	printf( "Received %d bytes from rank %d. Error code is %d. Number of threads is %d\n", count, c1, error, buf_rankData->openmp_numthreads );
                }
#endif

        		for ( int c2 = 0; c2 < buf_rankData->openmp_numthreads; c2++ ) {
        			buf_rankData->p_IObuf[(c2+1)*OUTPUT_LINE_LENGTH-1] = '\0';  // To be sure that no long lines are printed, though that would already be a bug.
        			printf( "%s\n", buf_rankData->p_IObuf + c2*OUTPUT_LINE_LENGTH );
        		}

        	}  // end for ( int c1 = 0; c1 < size; c1++ )

        	// Free the receive buffer
        	free( (void *) buf_rankData );

        }  // end if ( rank == 0 )

        // Make sure all data is send before clearing the buffer.
        MPI_Wait( &request, &status );

        // Free the send buffer
        free( (void *) my_rankData );

	} // end else part if ( unsorted_print )

    MPI_Barrier( MPI_COMM_WORLD );
	MPI_Finalize();

	return 0;
}

