/*
    This file is part of corona-13.

    corona-13 is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    corona-13 is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with corona-13. If not, see <http://www.gnu.org/licenses/>.
*/

#include "corona_common.h"
#include "pointsampler.h"
#include "sampler.h"
#include "render.h"
#include "pathspace.h"
#include "points.h"
#include "threads.h"
#include "ext/halton/halton.h"
#include "dbor.h"
#include "../tools/img/gaussconv"

#include <stdio.h>
#include <float.h>
#include <pthread.h>

//Number of Samples that should be sampled (0 means no stopping time)
static uint64_t RENDERSTOPTIME = 0;


//Define whether the resulting variance map should be convoluted using a gaussian filter
#define USEGAUSSIAN 0
//The Sigma Parameter for the gaussian filter
#define GAUSSIANSIGMA 1
//Define if the sample distribution should be printed out after every iteration (0 means no)
static int WRITEFBINDIVIDUAL = 0;
//Size of a single grid cell
#define GRID_SIZE 4
//Samples per Pixel standard value
#define SAMPLE_SIZE 1
//Multiplier that applies once sufficient Fireflies were found in a grid cell
#define FIREFLYMULTIPLIER 20
//Threshold that determines if enough fireflies were found to enhance sampling
#define FIREFLYTHRESHOLD 4
//Threshold to define when too many fireflies were found in a cell (Light sources)
#define MAX_BRIGHT_OUTLIERS 4
#define LIGHTSOURCEMULTIPLIER -1
static int init = 0;
static int HORIZONTALSIZE;
static int VERTICALSIZE;
static int gridnumber = 0;
static int spp = 0;
static int row = 0;
static int col = 0;
//number of horizontal grid cells
static int num_horizontal_cells;
//number of vertical grid cells
static int num_vertical_cells;
//contains the factors that are currently used for adaptive sampling
static int *sample_factor;
//contains the factors that are used in the next iteration
static int *new_factor;
//contains the amount of found fireflies for each grid cell
static int *num_fireflies;
static int *num_bright_outliers;
//number of iteration, important for naming the .pfm files after each
static int runNumber = 0;
//Contains the sample distribution over the whole scene
static float *overallSamples;
pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
static float averageSPP = 0;
static uint64_t allSamplesValue = 0;

#define MAXTRUST 40
#define MINTRUST 0.003
#define MAXSAMPLES 20

typedef struct fake_randoms_t
{
  int enabled;
  float rand[40];
}
fake_randoms_t;

typedef struct pointsampler_t
{
  fake_randoms_t *rand;
  halton_t h;
  uint64_t reinit;
}
pointsampler_t;

void pointsampler_print_info(FILE *f)
{
  fprintf(f, "mutations: halton points\n");
}

pointsampler_t *pointsampler_init(uint64_t frame)
{
  pointsampler_t *s = (pointsampler_t *)malloc(sizeof(pointsampler_t));
  s->rand = calloc(rt.num_threads, sizeof(*s->rand));
  s->reinit = 0;
  halton_init_random(&s->h, frame);
  return s;
}

int pointsampler_accept(path_t *curr, path_t *tent) { return 0; }
void pointsampler_clear() {}
void pointsampler_cleanup(pointsampler_t *s)
{
  free(s->rand);
  free(s);
}
void pointsampler_set_large_step(pointsampler_t *t, float p_large_step) {}
void pointsampler_finalize(pointsampler_t *s) {}

float pointsampler(path_t *p, int i)
{
  const int tid = common_get_threadid();
  if(rt.pointsampler->rand[tid].enabled)
    return rt.pointsampler->rand[tid].rand[i];

  int v = p->length;
  const int end = p->v[v].rand_beg;
  const int dim = end + i;
  if(dim >= halton_get_num_dimensions())
    // degenerate to pure random mersenne twister
    return points_rand(rt.points, common_get_threadid());
  else
    // note that this clips the bits in p->index to 32:
    return halton_sample(&rt.pointsampler->h, dim, p->index);
}

void pointsampler_splat(path_t *p, mf_t value)
{
  render_splat(p, value);
}

int getFactor(float i, float j) {
  int factor = 1;
  //Calculate the factor for the Gridcell, for normalizing
  int row = floor(i / GRID_SIZE);
  int col = floor(j / GRID_SIZE);
  factor = sample_factor[col * num_horizontal_cells + row];
  return factor;
}

uint64_t getSampleCount(float width, float height) {
    if (init == 0 || runNumber == 0) {
    return width * height;
  }
  uint64_t sampleSum = 0;
  for (int i = 0; i < width / GRID_SIZE * height / GRID_SIZE; i++) {
    if (!sample_factor[i]) sampleSum += 1; else {
      sampleSum += sample_factor[i];
    }
  }
  sampleSum = sampleSum * GRID_SIZE * GRID_SIZE;
  return sampleSum;
}

void setNewFactor(int dborLevel, float i, float j) {
  int row = floor(i / GRID_SIZE);
  int col = floor(j / GRID_SIZE);
  //If a very bright spot was found (eventually a light source)
  if (dborLevel == LIGHTSOURCEMULTIPLIER) {
    num_bright_outliers[col* num_horizontal_cells + row]++;
  }
  if (num_bright_outliers[col* num_horizontal_cells + row] >= MAX_BRIGHT_OUTLIERS) {
    //light source was found, do not sample cell often
 //  printf("Light source was reduced at pixel %f and %f\n", i, j);
    new_factor[col * num_horizontal_cells + row] = 1;
    return;
  }

  //If a firefly was found
  if (dborLevel == FIREFLYMULTIPLIER) {
    //Only set the factor high when a sufficient amount of fireflies was found in the grid cell
    if (num_fireflies[col * num_horizontal_cells + row] < FIREFLYTHRESHOLD) {
      num_fireflies[col * num_horizontal_cells + row]++;
    } else {
      new_factor[col * num_horizontal_cells + row] = FIREFLYMULTIPLIER;
    }
  //Otherwise set the factor to the highest dbor level
  } else if (new_factor[col * num_horizontal_cells + row] < dborLevel) {
    new_factor[col * num_horizontal_cells + row] = dborLevel;
  }
}

int translateTrustToSampleValue(float trust) {
  if (trust >= MAXTRUST) {
   // printf("trust was %f and set to 1\n", trust);
    return 1;
  } else if (trust <= MINTRUST) {
    return MAXSAMPLES;
   // printf("trust value was %f and set to 20\n", trust);
  } else {
    // Linear graph that declines in value with higher trust
    float m = MINTRUST - MAXSAMPLES / (MAXTRUST - MINTRUST);
   // printf("trust value was %f\n", floor(m * trust + MAXSAMPLES - m));
    return floor(m * trust + MAXSAMPLES - m);
  }
  return 1;
}

void setFactorTrust(int value, int row, int col) {
  new_factor[col * num_horizontal_cells + row] = value;
  //printf("Set factor of row %d and col %d to %d \n", row, col, value);
}

void pointsampler_mutate(path_t *curr, path_t *tent)
{
  if (init == 0) {
    runNumber = 0;
    HORIZONTALSIZE = view_width();
    VERTICALSIZE = view_height();

    num_horizontal_cells = (HORIZONTALSIZE / GRID_SIZE);
    num_vertical_cells = (VERTICALSIZE / GRID_SIZE);
    //Initialize all the framebuffers
    int *factors;
    int *newFactors;
    int *fireflies;
    int *brightOutliers;
    factors = (int *) malloc(num_horizontal_cells*num_vertical_cells * sizeof(int));
    newFactors = (int *) malloc(num_horizontal_cells*num_vertical_cells * sizeof(int));
    fireflies = (int *) malloc(num_horizontal_cells*num_vertical_cells * sizeof(int));
    brightOutliers = (int *) malloc(num_horizontal_cells*num_vertical_cells * sizeof(int));

    //Initialize framebuffer where the samples are saved
    overallSamples = (float *) malloc(num_horizontal_cells*num_vertical_cells * sizeof(float));

    for (int i = 0; i < num_horizontal_cells * num_vertical_cells; i++) {
      factors[i] = 1;
      newFactors[i] = 1;
      fireflies[i] = 0;
      overallSamples[i] = 0;
      brightOutliers[i] = 0;
    }
    new_factor = newFactors;
    sample_factor = factors;
    num_fireflies = fireflies;
    num_bright_outliers = brightOutliers;
    factors = NULL;
    fireflies = NULL;
    newFactors = NULL;
    brightOutliers = NULL;
    init = 1;
  }
  //Calculate random value within given pixel and mutate with that value
  double i = points_rand(rt.points, common_get_threadid());
  i = i + row + (gridnumber % num_horizontal_cells) * GRID_SIZE;
  double j = points_rand(rt.points, common_get_threadid());
  j = j + col + (gridnumber / num_horizontal_cells) * GRID_SIZE;
  pointsampler_mutate_with_pixel(curr, tent, i, j);

  pthread_mutex_lock(&mutex);
  ++spp;
  //Change to new Pixel, row gets incremented first                 //So the first run does not go through 3 times
  if (spp >= sample_factor[gridnumber] * SAMPLE_SIZE || (runNumber == 0 && spp >= sample_factor[gridnumber] * SAMPLE_SIZE)) {
    spp = 0;
    ++row;
    //Change to new Pixel, now new column
    if (row >= GRID_SIZE) {
      ++col;
      row = 0;
      //Change to new grid cell
      if (col >= GRID_SIZE) {
        col = 0;
        ++gridnumber;
        //Change into first grid cell since all grid cells have been sampled
        if (gridnumber >= num_horizontal_cells * num_vertical_cells) {
              if (MAPTRUSTTOLEVEL == 1 && runNumber != 0) {
                for (int x = 0; x < num_horizontal_cells; x++) {
                  for (int y = 0; y < num_vertical_cells; y++) {
                    float averageValue = 0.0f;
                    for (int r = 0; r < GRID_SIZE; r++) {
                      for (int c = 0; c < GRID_SIZE; c++) {
                        //Get accumulated trust value for the highest dbor levels of the pixel and adjust the samples according to that
                        averageValue += getTrustOfHighestDBORLevel(x * GRID_SIZE + r, y * GRID_SIZE + c);
                      }
                    }
                    averageValue = averageValue / (GRID_SIZE * GRID_SIZE);
                    if (averageValue >= 0.0001) {
                      int value = translateTrustToSampleValue(averageValue);
                      //printf("Average Value was %f\n", averageValue);
                      setFactorTrust(value, x, y);
                    }
                  }
                }
              }
          //New factors are now taken into account
          sample_factor = new_factor;
          if (USEGAUSSIAN == 1) {
            applygaussian();
          }
          //Write out current sample distribution as a .pfm file
          write_samples_as_framebuffer();
          resetNewFactor();
          gridnumber = 0;
          ++runNumber;
        }
      }
    }
  }
  pthread_mutex_unlock(&mutex);
}

void write_samples_as_framebuffer() {
  framebuffer_t *fb = malloc(sizeof(framebuffer_t));
  framebuffer_header_t *fb_header = malloc(sizeof(framebuffer_header_t));

  fb_header->channels = 3;
  fb_header->gain = 1;
  fb_header->height = num_vertical_cells;
  fb_header->width = num_horizontal_cells;

  fb->header = fb_header;
  fb->retain = 0;

  float* buffer = (float *) malloc(3 * num_horizontal_cells*num_vertical_cells * sizeof(float));
  float maxValue = 0.0f;
  int spp = 0;
  for (int i = 0; i < num_horizontal_cells * num_vertical_cells; i++) {
    //Divide by the highest possible Value to clamp the resulting values between 0 and 1
    buffer[3*i] = (sample_factor[i] + 0.0f) / FIREFLYMULTIPLIER;
    buffer[3*i+1] = (sample_factor[i] + 0.0f) / FIREFLYMULTIPLIER;
    buffer[3*i+2] = (sample_factor[i] + 0.0f) / FIREFLYMULTIPLIER;

    overallSamples[i] = overallSamples[i] + sample_factor[i];
    //find the highest number and divide by it to clamp the values between 0 and 1
    if (maxValue < overallSamples[i]) { maxValue = overallSamples[i]; }
    spp += sample_factor[i];
  }
  spp = spp * SAMPLE_SIZE;
  allSamplesValue += spp * GRID_SIZE * GRID_SIZE;
  averageSPP = averageSPP + spp / (num_horizontal_cells * num_vertical_cells);
 // printf("AverageSPP is %f and OverallSamples is %lu\n", averageSPP, allSamplesValue);
  fb->fb = buffer;

  framebuffer_t *overallSamplesFb = malloc(sizeof(framebuffer_t));
  overallSamplesFb->header = fb_header;
  overallSamplesFb->retain = 0;
  float* overallBuffer = (float *) malloc(num_horizontal_cells*num_vertical_cells * sizeof(float));
  for (int i = 0; i < num_horizontal_cells * num_vertical_cells; i++) {
      overallBuffer[i] = overallSamples[i] / (maxValue + 0.0f);
  }
  overallSamplesFb->fb = buffer;
  fb_export(overallSamplesFb, "OverallSamples.pfm", 0, 0);
  
  char str[100] = "";
  sprintf(str, "%d", runNumber);
  char* name = strcat(str, "Samples.pfm");
  if (WRITEFBINDIVIDUAL == 1) {
    fb_export(fb, name, 0, 0);
  }

  if (RENDERSTOPTIME != 0 && allSamplesValue >= RENDERSTOPTIME) {
    screenshotAndStop(allSamplesValue);
    allSamplesValue = 0;
  }
}

void applygaussian() {
    float* sampleFactorFloat = malloc(sizeof(float)*num_horizontal_cells*num_vertical_cells);
    for(int i = 0; i < num_vertical_cells*num_horizontal_cells; i++) {
      sampleFactorFloat[i] = sample_factor[i] + 0.0f;
    }
    iir_gauss_blur(num_horizontal_cells, num_vertical_cells, 1, sampleFactorFloat, GAUSSIANSIGMA);
    for(int i = 0; i < num_vertical_cells*num_horizontal_cells; i++) {
      if ((sampleFactorFloat[i] - floor(sampleFactorFloat[i])) < 0.5) {
        sample_factor[i] = floor(sampleFactorFloat[i]);
      } else {
        sample_factor[i] = ceil(sampleFactorFloat[i]);
      }
    }
}

void resetNewFactor() {
    for (int i = 0; i < num_horizontal_cells * num_vertical_cells; i++) {
      num_fireflies[i] = 0;
      new_factor[i] = 1;
      num_bright_outliers[i] = 0;
    }

}

void pointsampler_mutate_with_pixel(path_t *curr, path_t *tent, float i, float j)
{
  path_init(tent, tent->index, tent->sensor.camid);
  path_set_pixel(tent, i, j);
  sampler_create_path(tent);
}

void pointsampler_enable_fake_random(pointsampler_t *s)
{
  const int tid = common_get_threadid();
  s->rand[tid].enabled = 1;
}

void pointsampler_disable_fake_random(pointsampler_t *s)
{
  const int tid = common_get_threadid();
  s->rand[tid].enabled = 0;
}

void pointsampler_set_fake_random(pointsampler_t *s, int dim, float rand)
{
  const int tid = common_get_threadid();
  s->rand[tid].rand[dim] = rand;
}

void pointsampler_prepare_frame(pointsampler_t *s)
{
  // would wrap around int limit and run out of bits for radical inverse?  stop
  // to re-init the random bit permutations (we only pass a seed, will be
  // randomised internally):
  if(rt.threads->end >> 32 > s->reinit)
    halton_init_random(&s->h, rt.anim_frame + ++s->reinit);
}

void pointsampler_stop_learning(pointsampler_t *s) { }
