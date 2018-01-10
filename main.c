#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "stdio.h"
#include "evmdm6437.h"
#include "evmdm6437_dip.h"
#include "evmdm6437_led.h"
#include "conv1_weightsd.txt"
#include "conv2_weightsd.txt"
#include "fc1_wd.txt"
#include "fc2_wd.txt"
#include "conv1_bd.txt"
#include "conv2_bd.txt"
#include "fc1_bd.txt"
#include "fc2_bd.txt"
#include "display_datad.txt"
#include "test_datad.txt"
extern Int16 video_loopback_test();

// size for buffer_in: 720 * 480 / 2, the reason is explained below. 
#define Pixels 172800

// Resolution 720 * 480 (NTSC mode)
#define vWidth 720
#define vHeight 480
#define BufferNum 30


float y[32*8*8];
float output1[32*32*16];
float temp_array1[16*16*16];
float evidence1 [128];
float output2[32*32*16*32];
float temp_array2[8*8*32];
float evidence2[63];
float reshape_img[32*32];
float input_padded1[36*36];
float input_padded2[20*20*16];
float softmax_r[63];
float temp_img[64*64];

Uint32 tmp_char_display[16*33];
Int32 test_num = 945;
int thershold;

Int32 loc_x[9] = {4, 76,148,220,292,364,436,508,580};
Int32 loc_x1[9] = {2,38,74,110,146,182,218,254,290};
Int32 loc_y[6] = {4,76,148,220,292,364};

//Uint32 display_chars[62][64*64/2];

// Define a space on memory for save the information input and output (Interface data)
Uint32 buffer_out[Pixels]; //from 0x80000000
Uint32 buffer_in[Pixels]; //from 0x800A8C00, which is the same as 4 (bytes for integer) * Pixels

// Intermediate buffer to hold RGB output
Uint8 Buffer_input[vWidth*vHeight];
Uint32 bw[256];
int thershold_percent1 = 64*64*0.18;
int thershold_percent2 = 64*64*0.16;
int thershold_percent3 = 64*64*0.14;

// Define internal buffer
Uint32 internal_buffer1[Pixels / BufferNum];
Uint8 internal_buffer2[2 * Pixels / BufferNum *3];

// Define the position of the data (refer to linker.cmd)
// Internal memory L2RAM ".l2ram" 
// External memory DDR2 ".ddr2"
#pragma DATA_SECTION(buffer_out,".ddr2")
#pragma DATA_SECTION(buffer_in,".ddr2")
#pragma DATA_SECTION(Buffer_input, ".user")
#pragma DATA_SECTION(display_chars, ".user")
#pragma DATA_SECTION(tmp_char_display, ".user")
#pragma DATA_SECTION(test_data_dsp, ".user")
#pragma DATA_SECTION(conv1_weights, ".user")
#pragma DATA_SECTION(conv1_bias, ".user")
#pragma DATA_SECTION(conv2_weights, ".user")
#pragma DATA_SECTION(conv2_bias, ".user")
#pragma DATA_SECTION(fc1_weights, ".user")
#pragma DATA_SECTION(fc1_bias, ".user")
#pragma DATA_SECTION(fc2_weights, ".user")
#pragma DATA_SECTION(fc2_bias, ".user")
#pragma DATA_SECTION(y, ".user")
#pragma DATA_SECTION(output1, ".user")
#pragma DATA_SECTION(temp_array1, ".user")
#pragma DATA_SECTION(evidence1, ".user")
#pragma DATA_SECTION(output2, ".user")
#pragma DATA_SECTION(temp_array2, ".user")
#pragma DATA_SECTION(evidence2, ".user")
#pragma DATA_SECTION(reshape_img, ".user")
#pragma DATA_SECTION(input_padded1, ".user")
#pragma DATA_SECTION(input_padded2, ".user")
#pragma DATA_SECTION(temp_img, ".user")

// buffer_in represents one input frame which consists of two interleaved frames.
// Each 32 bit data has the information for two adjacent pixels in a row.
// Thus, the buffer holds 720/2 integer data points for each row of 2D image and there exist 480 rows.
//
// Format: yCbCr422 ( y1 | Cr | y0 | Cb )
// Each of y1, Cr, y0, Cb has 8 bits
// For each pixel in the frame, it has y, Cb, Cr components
//
// You can generate a lookup table for color conversion if you want to convert to different color space such as RGB.
// Could refer to http://www.fourcc.org/fccyvrgb.php for conversion between yCbCr and RGB
// 


// Copy data from input buffer to output buffer
// Format: yCbCr422 ( y1 | Cr | y0 | Cb )
// Each of y1, Cr, y0, Cb has 8 bits
void ycbcr2grey(void){
	int i,j;
	Uint32 temp;
	Uint8 y1;
	Uint8 y0;
    j=0;
	for(i = 0; i <Pixels; ++i){
		temp=buffer_out[i];
		y1 = (temp & 0xFF000000)>>24;
		y0 = (temp & 0x0000FF00)>>8;	
		Buffer_input[j] =(1.16388*(y0));
		Buffer_input[j+1] = (1.16388*(y1));
		j=j+2;
	}
}

void grey(float camera_cc[]){
	int i = 0;
	int j = 0;
	for(i =0;i<Pixels;i++){
		
		Uint32 temp1 = (((Uint8)(camera_cc[j]*255/1.16388))<<8) & 0x0000FF00;
		Uint32 temp2 = (((Uint8)(camera_cc[j+1]*255/1.16388))<<24)& 0xFF000000;
		buffer_out[i] = temp1 |temp2;
		j=j+2;
	}
}
void print_grid(void){
	Uint32 temp;
	Int32 i,j,k;
	i = 0;
	j=0;
	k=0;
	do{	
		if((i%36)==0){buffer_out[i]=0;}
		else{buffer_out[i]=buffer_in[i];}
		i++;	 
	} while (i < Pixels);
	do{
		for (j=0; j<vWidth; ++j) {buffer_out[k + j] = 0;}
		k = k + vWidth*36;
	} while (k < Pixels);
}

void output_input_video(void){
	Int32 i;
	i = 0;
	do{	
		if(i%(36*vWidth)==0){i = i + vWidth;}
		if((i%36)!=0 ){buffer_out[i]=buffer_in[i];}
		i++;	 
	} while (i < Pixels);
}



void array_reshape(float x[],float y_a[], int D_in, int D_out, int H, int W){
    int i = 0;
    int j = 0;
    int k = 0;
    int z = 0;
    for(i =0;i<D_out;i++){
        for( j =0;j<D_in;j++){
            for( k =0;k<H;k++){
                for( z =0;z<W;z++){
                    y_a[i*(D_in)+j+(D_in*D_out)*z+k*(D_in*D_out)*W] = x[z+k*W+j*W*H+i*W*H*D_in];
                }
            }
        }
    }
}


void replace_segment_input(Uint32 origional_array[], Uint32 replace_img_array[], int camera_image_width_05, int 

	camera_image_height, int segment_width_05, int segment_height, int location_x, int location_y){
	    
	    int temp_pos;
	    int size_of_replace = segment_height*segment_width_05;
	    
	    int k =0;
	    int counter_x = 0;
	    int counter_y = 0;
	    do{
	        temp_pos = location_x+(location_y+counter_y)*camera_image_width_05 +counter_x;
	        origional_array[temp_pos] = replace_img_array[counter_x+counter_y*segment_width_05];
	        counter_x++;
	        if(counter_x == segment_width_05){
	            counter_x = 0;
	            counter_y++;
	        }
	        k++;
	    }while( k < (size_of_replace));
}


void reshape_img_half(float segmented_input[],float half_ed[],int dim_w, int dim_h){
    int i = 0;
    int j = 0;
    int k = 0;
    for (k = 0; k < dim_w*dim_h/4; k++){
        half_ed[k] = (segmented_input[i*dim_w+j] + segmented_input[(i+1)*dim_w+j] + segmented_input[i*dim_w+j+1] + segmented_input[(i+1)*dim_w+j+1])/4;
        j = j + 2;
        if ( j > dim_w-2){
            i = i +2;
            j = 0;
        }
    }
}

void conv2d_relu( float input[],float input_padded[], float output_conv[], float weights[], float biases[], int K, int F, int S, int input_h, int input_w, int input_D){
    int H_in = input_h;
    int W_in = input_w;
    int D_in = input_D;
    int P = ((W_in-1)*S -W_in +F)/2;
    int D_out=K;
    
    int  i = 0;
    for(i =0; i<(W_in+2*P)*(H_in+2*P)*D_in; i++){
        input_padded[i] = 0.0;
    }
    
    int temp_pos;
    int j = 0;
    for ( j =0;j<D_in;j++){
        int counter_x = 0;
        int counter_y = 0;
        int k =j*W_in*H_in;
        do{
            temp_pos = P+(P+counter_y)*(W_in+2*P) + counter_x + j*(W_in+2*P)*(H_in+2*P);
            input_padded[temp_pos] = input[k];
            counter_x++;
            if(counter_x == W_in){
                counter_x = 0;
                counter_y++;
            }
            k++;
        }while( k < (j+1)*(W_in*H_in));
    }
    
 
    
    i = 0;
    j = 0;
    int d = 0;
    temp_pos = 0;
    int pointer_out = 0;
    for ( d = 0; d< D_out; d++){
        for ( j = 0; j<= 2*P+H_in-F; j = j+S){
            for ( i = 0 ; i <= 2*P+W_in-F; i=i+S){
                int counter_x = 0;
                int counter_y = 0;
                int counter_z = 0;
                int k = 0;
                float sum = 0;
                do{
                    temp_pos = i+counter_x+(j+counter_y)*(W_in+2*P)+ (counter_z)*(W_in+2*P)*(H_in+2*P);
                    sum = sum + input_padded[temp_pos]*weights[d*F*F*D_in+k];
                    
                    counter_x++;
                    if(counter_x == F){
                        counter_x = 0;
                        counter_y++;
                    }
                    if(counter_y==F){
                        counter_y=0;
                        counter_z++;
                    }
                    k++;
                }while( k < (F*F)*D_in);
                output_conv[pointer_out] = sum +biases[d];
                if(output_conv[pointer_out]<0){
                    output_conv[pointer_out] = 0;
                }
                
                pointer_out++;
            }
        }
    }
}

void max_pool(float input[], float max_pool[], int dim_w, int dim_h, int dim_d){
    int i = 0;
    int j = 0;
    int z = 0;
    int k = 0;
    
    for (k = 0; k < dim_d*dim_w*dim_h/4; k++){
        float local_tmp = input[i+j*dim_w+z*(dim_h*dim_w)];

        if(input[(i+1)+j*dim_w+z*(dim_h*dim_w)] > local_tmp){
            local_tmp = input[(i+1)+j*dim_w+z*(dim_h*dim_w)] ;
        }
        if(input[i+(j+1)*dim_w+z*(dim_h*dim_w)] > local_tmp){
            local_tmp = input[i+(j+1)*dim_w+z*(dim_h*dim_w)];
        }
        if(input[(i+1)+(j+1)*dim_w+z*(dim_h*dim_w)] > local_tmp){
            local_tmp = input[(i+1)+(j+1)*dim_w+z*(dim_h*dim_w)];
        }
        max_pool[k] = local_tmp;     

        i = i + 2;
        if ( i > dim_w-2){
            j = j +2;
            i = 0;
        }
        if ( j > dim_h-2){
            j =0;
            z++;
        } 
    }

}


void fc(float w[], float b[],float evidence[],float x[],int middle_layer, int input_num){
    int j = 0;
    int i = 0;
    for (j = 0; j < middle_layer; j++) {
        float tmp1 = 0 ;
        for (i = 0; i < input_num; i++) {
            tmp1 = tmp1 + w[j+i*middle_layer]*x[i];
        }
        evidence[j] = tmp1 + b[j];
    }
}


int softmax(float evidence_s[],int class_num){
    int k, h, m;
    double tmp2 = 0;
    double tmp3 = 0;
    int result = 0;
    
    for (k = 0; k < class_num; k++) {
        tmp2 = tmp2 + exp(evidence_s[k]);
        
    }
    //printf("temp2: %lf",tmp2);
    //printf("y %f\n",tmp2);
    for (m = 0; m < class_num; m++) {
        softmax_r[m] = exp(evidence_s[m]) / tmp2 ;
        //printf("%lf \n",softmax_r[m]);
        //printf("y %f\n",(evidence[m]));
    }
    for (h = 0; h < class_num; h++) {
        if (softmax_r[h]>tmp3){
            result = h;
            tmp3 = softmax_r[h];
        }
    }
    return result;
}

int segment_input(Uint8 camera_input[], int camera_image_width, int camera_image_height, int segment_width, int segment_height, int i, int j, int thershold_percent){
    //information from Buffer_input in gray scale with grid size 60*60   
    int temp_pos;
    int k =0;
    int counter_x = 0;
    int counter_y = 0;
    do{
        temp_pos = loc_x[i]+counter_x+(loc_y[j]+counter_y)*camera_image_width;
		temp_img[k] = camera_input[temp_pos];
        counter_x++;
        if(counter_x == segment_width){
            counter_x = 0;
            counter_y++;
        }
        k++;
    }while( k < (segment_width*segment_height));
	int is = 0;
	for(is=0;is<255;is++){
		bw[is]=0;
	}
	is=0;
	for(is=0;is<64*64;is++){
		int temp = temp_img[is];
		bw[temp]++;
	}
	int sum =0;
	is=0;
	for(is=0;is<255;is++){
		sum = sum + bw[is];
		if(sum > thershold_percent){
			thershold=is;
			break;
		}
	}
    
    int tmp_result1=0;

    reshape_img_half((temp_img),reshape_img,segment_width,segment_height);
    for(is =0;is<32*32;is++){
    	if( reshape_img[is] > thershold){
			reshape_img[is] = 0.999900;
		}
		else {
			reshape_img[is] = 0.000000;
		}
    }
        
    int ic = 0;
	int jc = 0;
	Uint32 temp1 ;
	Uint32 temp2;
	for(ic =0;ic<16*32;ic++){
		
		temp1= (((Uint8)(reshape_img[jc]*255/1.16388))<<8) & 0x0000FF00;
		temp2= (((Uint8)(reshape_img[jc+1]*255/1.16388))<<24)& 0xFF000000;
		tmp_char_display[ic] = temp1 | temp2;
		jc=jc+2;
	}
	replace_segment_input(buffer_out,tmp_char_display ,vWidth/2,vHeight,16,32,loc_x1[i],loc_y[j]);
	
    conv2d_relu(reshape_img,input_padded1,output1, conv1_weights, conv1_bias, 16, 5, 1, 32, 32, 1);
    
    max_pool(output1,temp_array1, 32, 32, 16);
    
    conv2d_relu(temp_array1,input_padded2, output2, conv2_weights, conv2_bias, 32, 5, 1, 16, 16, 16);
    
    max_pool(output2,temp_array2, 16, 16, 32);
    
    array_reshape(temp_array2,y, 32, 1, 8, 8);
    
    fc(fc1_weights,fc1_bias,evidence1,y ,128, 2048);
    
    fc(fc2_weights, fc2_bias,evidence2, evidence1, 63, 128);
    
    tmp_result1 = softmax(evidence2, 63);
    printf("temp_result: %d\n",tmp_result1);
    
    return tmp_result1;
}

void main( void )
{
	Int16 dip0, dip1, dip2, dip3;

	/* Initialize BSL */
	EVMDM6437_init();
	
    /* Initialize the DIP Switches & LEDs if needed */
    EVMDM6437_DIP_init( );
    EVMDM6437_LED_init( );
    
	// Initialize video input/output 
	video_loopback_test();
	print_grid();
    //read_display();
	while (1){		
        /* Will return DIP_DOWN or DIP_UP */
        dip0 = EVMDM6437_DIP_get( DIP_0 );
        dip1 = EVMDM6437_DIP_get( DIP_1 );
        dip2 = EVMDM6437_DIP_get( DIP_2 );
		output_input_video();
		int i =0;
		int j = 0;
        // Run different procedures depending on the DIP switches pressed.
        if(dip0 == DIP_DOWN){
        	ycbcr2grey();
        	//grey();
	        for(j=0;j<6;j++){
	       		for(i=0;i<9;i++){
	            	int r1 = segment_input(Buffer_input, vWidth, vHeight, 64, 64, i, j,thershold_percent1);
	            	int r2 = segment_input(Buffer_input, vWidth, vHeight, 64, 64, i, j,thershold_percent2);
	            	int r3 = segment_input(Buffer_input, vWidth, vHeight, 64, 64, i, j,thershold_percent3);
	            	int vote = 0;
	            	int pr12=0;
	            	int pr23=0 ;
	            	int pr13=0;
	            	if(r1 ==r2 ){vote++;pr12 = 1;}
	            	if(r1 == r3){vote++;pr13 = 1;}
	            	if(r2==r3){vote++;pr23 = 1;}
	            	int tmp_result1;
	            	if(vote==0){tmp_result1 = r2;}
	            	else{
	            		if(pr12==1){tmp_result1 = r1;}
	            		if(pr23 == 1){tmp_result1 = r2;}
	            		if(pr13 ==1){tmp_result1 = r3;}
	            	}
	            	printf("\n");
	            	replace_segment_input(buffer_out,display_chars[tmp_result1],vWidth/2,vHeight,64/2,64,loc_x1[i],loc_y[j]);
	            	dip1 = EVMDM6437_DIP_get( DIP_1 );
	            	if(dip1 == DIP_DOWN){
						break;
					}
					dip2 = EVMDM6437_DIP_get( DIP_2 );
					do{
						dip2 = EVMDM6437_DIP_get( DIP_2 );
					} while (dip2 != DIP_DOWN);
	       		}
	       		dip1 = EVMDM6437_DIP_get( DIP_1 );
       			if(dip1 == DIP_DOWN){
					j=0;
					for(j =0;j<1000000;j++){}
					break;
				}
	        }
        }
	}
}
