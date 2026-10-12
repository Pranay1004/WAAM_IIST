#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>
#include <mpi.h>
// Binary alloy Ni and Cu
int main(int argc, char* argv[]) {

    // Physical data (Ni)

    double sigma_A = 0.37;  // surface energy (j/m^2)
    double nu_A = 2e2;      // kinetic coefficient (m/s/K)
    double Tm_A = 1728;     // Melting Temp
    double L_A = 2.35e9;     // Latent heat (J/m^3)
    double Cp_A = 5.42e6;     // Heat capacity(J/m^3/k)
    double D_lA=1e-9;         // Diffusitivity in liq (m^2/s)
    double D_sA=1e-13;       // Diffusitivity in solid (m^2/s)


    // Physical data (Cu)

    double sigma_B = 0.29;
    double nu_B = 2.47e2;
    double Tm_B = 1358;
    double L_B = 1.728e9;
    double Cp_B = 3.96e6;
    double D_lB=1e-9;
    double D_sB=1e-13;



    int aniso = 4;
    double theta0 = 0;
    double lambda_A=4.9e-6/2;           //interface width
    double lambda_B=4.9e-6/2;
    // Define model parameters

    double w_A = 6.6 * sigma_A / lambda_A;            // free energy barrier
    double eps0_A = sqrt(2.73 * lambda_A * sigma_A);  // interface thickness / gradient energy coefficient
    double M_A = Tm_A * nu_A / 2.73 / L_A / lambda_A; // phase field mobility

    double w_B = 6.6 * sigma_B/ lambda_B;
    double eps0_B = sqrt(2.73 * lambda_B * sigma_B);
    double M_B = Tm_B * nu_B / 2.73 / L_B / lambda_B;

    double delta = 0.04;
    double alfa = 0.4;  // Noise
    double vm=7.42e-6;
    double R=8.314;
    // Define grid parameters
    int Nx = 512;
    int Ny = 512;

    double dx = lambda_A/4;
    double dy = dx;

    // Define time integration parameters
    double nstep = 1e6;
    double dt = dx*dx*.1;

    // Initialize nucleus
    //double phi[Nx+10][Ny+10], T[Nx+10][Ny+10],c[Nx+10][Ny+10];
    double** phi = (double**)malloc(Nx * sizeof(double));
  double** T = (double**)malloc(Nx * sizeof(double));
  double** c = (double**)malloc(Nx * sizeof(double));
   for (int i = 0; i < Nx; i++) {
        phi[i] = (double*)malloc(Ny * sizeof(double));
        T[i] = (double*)malloc(Ny * sizeof(double));
        c[i] = (double*)malloc(Ny * sizeof(double));
    }

    int r = 10; // radius of nucleus
    for (int i = 0; i < Nx; i++)
    {
        for (int j = 0; j < Ny; j++)
        {
            double distance = sqrt((i-Nx/2) * (i-Nx/2) + (j-Ny/2) * (j-Ny/2));

            if (distance < r)
                {
                phi[i][j] = 1.0;

                }

            T[i][j] = 1554.0;
            c[i][j]=.4083;
        }
    }

    // Evolve the system over time
    //double lap_phi[Nx + 10][Ny + 10],lap_c[Nx + 10][Ny + 10], lap_T[Nx + 10][Ny + 10], phi_dx[Nx + 10][Ny + 10], phi_dy[Nx + 10][Ny + 10], eps[Nx + 10][Ny + 10],d_eps[Nx + 10][Ny + 10];

    // Evolve the system over time
    double** lap_phi = (double**)malloc(Nx * sizeof(double));
   // double** lap_T = (double**)malloc(Nx * sizeof(double));
    double** lap_c = (double**)malloc(Nx * sizeof(double));
    double** phi_dx = (double**)malloc(Nx * sizeof(double));
    double** phi_dy = (double**)malloc(Nx * sizeof(double));
    double** eps = (double**)malloc(Nx * sizeof(double));
    double** d_eps = (double**)malloc(Nx * sizeof(double));
    for (int i = 0; i < Nx; i++) {
        lap_phi[i] = (double*)malloc(Ny * sizeof(double));
       // lap_T[i] = (double*)malloc(Ny * sizeof(double));
        lap_c[i] = (double*)malloc(Ny * sizeof(double));
        phi_dx[i] = (double*)malloc(Ny * sizeof(double));
        phi_dy[i] = (double*)malloc(Ny * sizeof(double));
        eps[i] = (double*)malloc(Ny * sizeof(double));
        d_eps[i] = (double*)malloc(Ny * sizeof(double));
    }



    double** phi1 = (double**)malloc(Nx * sizeof(double));
     double** T1 = (double**)malloc(Nx * sizeof(double));
     double** c1 = (double**)malloc(Nx * sizeof(double));
    double** lap_phi1 = (double**)malloc(Nx * sizeof(double));
   // double** lap_T1 = (double**)malloc(Nx * sizeof(double));
    double** lap_c1 = (double**)malloc(Nx * sizeof(double));
    double** phi_dx1 = (double**)malloc(Nx * sizeof(double));
    double** phi_dy1 = (double**)malloc(Nx * sizeof(double));
    double**  eps1= (double**)malloc(Nx * sizeof(double));
    double** d_eps1 = (double**)malloc(Nx * sizeof(double));
    for (int i = 0; i < Nx; i++) {

        phi1[i] = (double*)malloc(Ny * sizeof(double));
        T1[i] = (double*)malloc(Ny * sizeof(double));
        c1[i] = (double*)malloc(Ny * sizeof(double));
        lap_phi1[i] = (double*)malloc(Ny * sizeof(double));
      // lap_T1[i] = (double*)malloc(Ny * sizeof(double));
        lap_c1[i] = (double*)malloc(Ny * sizeof(double));
       phi_dx1[i] = (double*)malloc(Ny * sizeof(double));
       phi_dy1[i] = (double*)malloc(Ny * sizeof(double));
       eps1[i] = (double*)malloc(Ny * sizeof(double));
       d_eps1[i] = (double*)malloc(Ny * sizeof(double));

    }

    int n = 0;  //  Step Counter

    for (int i = 0; i < Nx; i++)
        {
            for (int j = 0; j < Ny; j++)
            {
                lap_phi[i][j] = 0.0;
               // lap_T[i][j] = 0.0;
                lap_c[i][j] = 0.0;
                phi_dx[i][j] = 0.0;
                phi_dy[i][j] = 0.0;
                eps[i][j] = 0.0;
                d_eps[i][j] = 0.0;
            }
        }
printf("Started");

clock_t start, end;
    /* Store start time here */
    start = clock();

    int rank, n_ranks, numbers_per_rank,nz;
 int my_first, my_last;
 int numbers = Nx;
 int a,b;

    // First call MPI_Init
 MPI_Init(&argc, &argv);
 // Get my rank and the number of ranks
 MPI_Comm_rank(MPI_COMM_WORLD,&rank);        //get my process id
 MPI_Comm_size(MPI_COMM_WORLD,&n_ranks);      //get number of processes
 MPI_Status status;

 // Calculate the number of iterations for each rank
 numbers_per_rank = floor(numbers/n_ranks);
 /*if( numbers%n_ranks > 0 ){
    // Add 1 in case the number of ranks doesn't divide the number of numbers
    numbers_per_rank += 1;
 }*/

 // Figure out the first and the last iteration for this rank
 my_first = rank * numbers_per_rank;
 my_last = my_first + numbers_per_rank-1;


//printf("My rank is : %d  , My first is %d , My last is :  %d \n" , rank,my_first,my_last);
  for (int iter = 1; iter <= nstep; iter++)
{

            if (rank==0)
            {
                for(int i=1; i<n_ranks;i++)
                {

                my_first = i * numbers_per_rank;
                my_last = my_first + numbers_per_rank-1;
                MPI_Send(&my_first, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
                MPI_Send(&my_last, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
                MPI_Send(&(phi[0][0]), Nx*Ny,MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
                MPI_Send(&(T[0][0]), Nx*Ny, MPI_DOUBLE,i, 0, MPI_COMM_WORLD);
                MPI_Send(&(c[0][0]), Nx*Ny, MPI_DOUBLE,i, 0, MPI_COMM_WORLD);
                MPI_Send(&(lap_phi[0][0]), Nx*Ny,MPI_DOUBLE, i ,0, MPI_COMM_WORLD);
               // MPI_Send(&(lap_T[0][0]), Nx*Ny, MPI_DOUBLE,i,  0, MPI_COMM_WORLD);
                MPI_Send(&(lap_c[0][0]), Nx*Ny, MPI_DOUBLE,i,  0, MPI_COMM_WORLD);
                MPI_Send(&(phi_dx[0][0]), Nx*Ny, MPI_DOUBLE,i,  0, MPI_COMM_WORLD);
                MPI_Send(&(phi_dy[0][0]), Nx*Ny, MPI_DOUBLE,i, 0, MPI_COMM_WORLD);
                MPI_Send(&(eps[0][0]), Nx*Ny, MPI_DOUBLE,i,  0, MPI_COMM_WORLD);
                MPI_Send(&(d_eps[0][0]), Nx*Ny, MPI_DOUBLE,i, 0, MPI_COMM_WORLD);



                }

                for(int i=1; i<n_ranks;i++)
                {
                MPI_Recv(&a, 1, MPI_INT, i,0, MPI_COMM_WORLD,&status);
                MPI_Recv(&b, 1, MPI_INT, i,0, MPI_COMM_WORLD,&status);
                MPI_Recv(&nz, 1, MPI_INT, i,0, MPI_COMM_WORLD,&status);
                MPI_Recv(&(phi1[0][0]), nz*Ny,MPI_DOUBLE, i, 0, MPI_COMM_WORLD,&status);
                MPI_Recv(&(T1[0][0]), nz*Ny, MPI_DOUBLE,i, 0, MPI_COMM_WORLD,&status);
                MPI_Recv(&(c1[0][0]), nz*Ny, MPI_DOUBLE,i, 0, MPI_COMM_WORLD,&status);
                MPI_Recv(&(lap_phi1[0][0]), nz*Ny,MPI_DOUBLE, i, 0, MPI_COMM_WORLD,&status);
               // MPI_Recv(&(lap_T1[0][0]), nz*Ny, MPI_DOUBLE,i, 0, MPI_COMM_WORLD,&status);
                 MPI_Recv(&(lap_c1[0][0]), nz*Ny, MPI_DOUBLE,i, 0, MPI_COMM_WORLD,&status);
                MPI_Recv(&(phi_dx1[0][0]), nz*Ny, MPI_DOUBLE,i, 0, MPI_COMM_WORLD,&status);
                MPI_Recv(&(phi_dy1[0][0]), nz*Ny, MPI_DOUBLE,i, 0, MPI_COMM_WORLD,&status);
                MPI_Recv(&(eps1[0][0]), nz*Ny, MPI_DOUBLE,i, 0, MPI_COMM_WORLD,&status);
                MPI_Recv(&(d_eps1[0][0]), nz*Ny, MPI_DOUBLE,i, 0, MPI_COMM_WORLD,&status);
                //printf("%d  %d %d,",a,b,nz);

                      for(int j=a;j<=b;j++)
                      {
                            for(int k=0;k<Ny;k++)

                            {

                               // phi[j][k]=phi1[j-a][k];
                                //T[j][k]=T1[j-a][k];
                                lap_phi[j][k]=lap_phi1[j-a][k];
                               // lap_T[j][k]=lap_T1[j-a][k];
                                 lap_c[j][k]=lap_c1[j-a][k];
                                phi_dx[j][k]=phi_dx1[j-a][k];
                                phi_dy[j][k]=phi_dy1[j-a][k];
                                eps[j][k]=eps1[j-a][k];
                                d_eps[j][k]=d_eps1[j-a][k];
                             }


                      }




                }

              for (int i = 0; i < Nx; i++) {
                for (int j = 0; j < Ny; j++) {
                    int ip = i + 1;
                    int im = i - 1;
                    int jp = j + 1;
                    int jm = j - 1;

                    if (im == -1) {
                        im = 0;
                    }
                    if (ip == Nx) {
                        ip = Nx - 1;
                    }
                    if (jm == -1) {
                        jm = 0;
                    }
                    if (jp == Ny) {
                        jp = Ny - 1;
                    }

                    // Time integration
                    double phi_old = phi[i][j];
                   double T_old = T[i][j];
                   double c_old = c[i][j];


                    double p=phi_old*phi_old*phi_old*(6*phi_old*phi_old+15*phi_old-10);

                    double D=D_sA+p*(D_lA-D_sA);
                    double M=(1-c[i][j])*M_A+c[i][j]*M_B;
                    //double K=(1-c[i][j])*K_A+c[i][j]*K_B;
                    //double Cp=(1-c[i][j])*Cp_A+c[i][j]*Cp_B;
                    double L= (1-c[i][j])*L_A+c[i][j]*L_B;


                    double t1 = (eps[i][jp]*d_eps[i][jp]*phi_dx[i][jp] - eps[i][jm]*d_eps[i][jm]*phi_dx[i][jm]) / 2 / dy;
                    double t2 = (eps[ip][j]*d_eps[ip][j]*phi_dy[ip][j] - eps[im][j]*d_eps[im][j]*phi_dy[im][j]) / 2 / dx;
                    double g = phi_old * phi_old * (1 - phi_old) * (1 - phi_old);
                    double g_dash = 2 * phi_old * (1 - phi_old) * (1 - phi_old) - 2 * phi_old * phi_old * (1 - phi_old);

                    double H_A=w_A*g_dash+30*g*L_A*(Tm_A-T[i][j])/Tm_A;
                    double H_B=w_B*g_dash+30*g*L_B*(Tm_B-T[i][j])/Tm_B;
                    double t3 =(1-c[i][j])*H_A+c[i][j]*H_B;

                    double random = -1 + (double)rand() / RAND_MAX * 2;
                    //double noise = 16 * alfa * random * phi_old * phi_old * (1 - phi_old) * (1 - phi_old);
                    double noise = -M*alfa*random*16*g*t3;


                    phi[i][j] = phi_old + (dt * M) * (t1 - t2 + eps[i][j] * eps[i][j] * lap_phi[i][j] - t3 + noise);
                    //T[i][j] = T_old + dt * K/Cp * lap_T[i][j] + L / Cp * 30 * g * (phi[i][j] - phi_old);
                    c[i][j]=c_old+dt*D*(lap_c[i][j]+vm/R*c[i][j]*(1-c[i][j])*(H_B-H_A)*lap_phi[i][j]);

                }

            }

            }
            else
            {
                MPI_Recv(&my_first, 1, MPI_INT,0, 0, MPI_COMM_WORLD,&status);
                MPI_Recv(&my_last, 1, MPI_INT,0, 0, MPI_COMM_WORLD,&status);
                MPI_Recv(&(phi[0][0]), Nx*Ny,MPI_DOUBLE,0, 0, MPI_COMM_WORLD,&status);
                MPI_Recv(&(T[0][0]), Nx*Ny, MPI_DOUBLE,0,0, MPI_COMM_WORLD,&status);
                MPI_Recv(&(c[0][0]), Nx*Ny, MPI_DOUBLE,0,0, MPI_COMM_WORLD,&status);
                MPI_Recv(&(lap_phi[0][0]), Nx*Ny,MPI_DOUBLE, 0, 0, MPI_COMM_WORLD,&status);
               // MPI_Recv(&(lap_T[0][0]), Nx*Ny, MPI_DOUBLE,0, 0, MPI_COMM_WORLD,&status);
                MPI_Recv(&(lap_c[0][0]), Nx*Ny, MPI_DOUBLE,0, 0, MPI_COMM_WORLD,&status);
                MPI_Recv(&(phi_dx[0][0]), Nx*Ny, MPI_DOUBLE,0, 0, MPI_COMM_WORLD,&status);
                MPI_Recv(&(phi_dy[0][0]), Nx*Ny, MPI_DOUBLE, 0,0, MPI_COMM_WORLD,&status);
                MPI_Recv(&(eps[0][0]), Nx*Ny, MPI_DOUBLE,0,0, MPI_COMM_WORLD,&status);
                MPI_Recv(&(d_eps[0][0]), Nx*Ny, MPI_DOUBLE,0, 0, MPI_COMM_WORLD,&status);
                //printf("%d  %d ,",my_first,my_last);
                for(int i = my_first; i < my_last; i++)
                    {
                            // printf("%i from thread number %i \n",i,omp_get_thread_num());
                           //#pragma omp parallel for
                            for(int j = 0; j < Ny; j++)
                            {


                                    int ip = i+1;
                    int im = i-1;
                    int jp = j+1;
                    int jm = j-1;

                    if(im == -1)
                    {
                        im = 0;
                    }
                    if(ip == Nx)
                    {
                        ip = Nx - 1;
                    }
                    if(jm == -1)
                    {
                        jm = 0;
                    }
                    if(jp == Ny)
                    {
                        jp = Ny - 1;
                    }

                    // Applying 5 point stencil
                    // Calculating Laplacian
                    double p1 = phi[ip][j]; // right
                    double p2 = phi[im][j]; // Left
                    double p3 = phi[i][jp]; // up
                    double p4 = phi[i][jm]; // down
                    double p5 = phi[i][j];  // centre

                    lap_phi[i-my_first][j] = (p1 + p2 + p3 + p4 - 4*p5) / (dx * dy);

                    p1 = T[ip][j]; // right
                    p2 = T[im][j]; // Left
                    p3 = T[i][jp]; // up
                    p4 = T[i][jm]; // down
                    p5 = T[i][j];  // centre

                  /*  lap_T[i-my_first][j] = (p1 + p2 + p3 + p4 - 4*p5) / (dx * dy);

                    p1 = c[ip][j]; // right
                    p2 = c[im][j]; // Left
                    p3 = c[i][jp]; // up
                    p4 = c[i][jm]; // down
                    p5 = c[i][j];  // centre*/

                    lap_c[i-my_first][j] = (p1 + p2 + p3 + p4 - 4*p5) / (dx * dy);

                    // Calculating gradient

                    phi_dx[i-my_first][j] = (phi[ip][j] - phi[im][j]) / (2 * dx);
                    phi_dy[i-my_first][j] = (phi[i][jp] - phi[i][jm]) / (2 * dy);

                    // Theta

                    double theta = atan2(phi_dy[i][j], phi_dx[i][j]);

                    // epsilon

                    eps[i-my_first][j] = eps0_A * (1 + delta * cos(aniso * (theta - theta0)));

                    // deps/dtheta

                    d_eps[i-my_first][j] = -eps0_A * delta * aniso * sin(aniso * (theta - theta0));
                            }
                  }

                //MPI_Send(&my_first, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
                //MPI_Send(&my_last, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
                 nz=my_last-my_first+1;

                MPI_Send(&my_first, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
                MPI_Send(&my_last, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
                MPI_Send(&nz, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
                MPI_Send(&(phi[0][0]), nz*Ny,MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
                MPI_Send(&(T[0][0]), nz*Ny, MPI_DOUBLE,0, 0, MPI_COMM_WORLD);
                MPI_Send(&(lap_phi[0][0]), nz*Ny,MPI_DOUBLE, 0 ,0, MPI_COMM_WORLD);
                //MPI_Send(&(lap_T[0][0]), nz*Ny, MPI_DOUBLE,0,  0, MPI_COMM_WORLD);
                MPI_Send(&(phi_dx[0][0]), nz*Ny, MPI_DOUBLE,0,  0, MPI_COMM_WORLD);
                MPI_Send(&(phi_dy[0][0]), nz*Ny, MPI_DOUBLE,0, 0, MPI_COMM_WORLD);
                MPI_Send(&(eps[0][0]), nz*Ny, MPI_DOUBLE,0,  0, MPI_COMM_WORLD);
                MPI_Send(&(d_eps[0][0]), nz*Ny, MPI_DOUBLE,0, 0, MPI_COMM_WORLD);



           }


            // Time Evolution
           //#pragma omp parallel for



}
//printf(".%d", iter);


                if(rank==0)
              {

                         end = clock();
                            /* Get the time taken by program to execute in seconds */
                            double duration = ((double)end - start)/CLOCKS_PER_SEC;

                            printf("Time taken to execute in seconds : %f", duration);


                            /*for (int i = 0; i < Nx; i++)
                            {
                                for (int j = 0; j < Ny; j++)
                                {
                                    printf("%e    ", phi[i][j]);
                                }
                                printf("\n");
                            }*/


                        char filename[9] = "phi.csv";

                            // open the file for writing
                            FILE *fp = fopen(filename, "w");

                            // write to the excel file
                            for (int i = 0; i < Nx; i++)
                            {
                                for (int j = 0; j < Ny; j++)
                                {
                                    fprintf(fp, "%e,    ", phi[i][j]);
                                }
                                fprintf(fp, "\n");
                            }

                            // close the file
                            fclose(fp);

                 }






                        for(int i = 0; i < Nx; i++)
                        {
                            free(phi[i]);
                            free(T[i]);
                             free(c[i]);
                            free(lap_phi[i]);
                            //free(lap_T[i]);
                            free(lap_c[i]);
                            free(phi_dx[i]);
                            free(phi_dy[i]);
                            free(eps[i]);
                            free(d_eps[i]);
                            }

                        free(phi);
                        free(T);
                         free(c);
                        free(lap_phi);
                       // free(lap_T);
                         free(lap_c);
                        free(phi_dx);
                        free(phi_dy);
                        free(eps);
                        free(d_eps);

 MPI_Finalize();
 }
