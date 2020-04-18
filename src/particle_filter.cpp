/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>
#include <cmath>
#include <stdlib.h>

#include "helper_functions.h"

using std::string;
using std::vector;
using std::normal_distribution;
using std::cout;
using std::endl;


void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
  */  
  
  std::default_random_engine gen;
  double std_x, std_y, std_theta;
  
  num_particles = 100;  //Set the number of particles
  
  // Assign the coordinates and theta
  x = x;
  y = y;
  theta =theta;
  
  // Get the standard deviations from the arguments
  std_x = std[0];
  std_y = std[1];
  std_theta = std[2];
  
  // Define normal distributions for use later 
  normal_distribution<double> dist_x(x, std_x);
  normal_distribution<double> dist_y(y, std_y);
  normal_distribution<double> dist_theta(theta, std_theta);
  
  // Loop to create the defined number of particles
  for (int i = 0; i < num_particles; ++i) {
    Particle particle;
    double sample_x, sample_y, sample_theta;
    
    // Initialize with normal distributions like this: 
    particle.id = i;
    particle.x = dist_x(gen);
    particle.y = dist_y(gen);
    particle.theta = dist_theta(gen);
    particle.weight = 1.0; 
    
    // Append the particle to the array
    particles.push_back(particle);
    
    // Append weights
    weights.push_back(1.0);
    
    // Print samples to the terminal.
    //std::cout << "Sample " << i + 1 << " " << sample_x << " " << sample_y << " " 
    //          << sample_theta << std::endl;        
  } 
  
  // set the flag to mark intialization as complete
  ParticleFilter::is_initialized = true;
  std::cout<<"Intialization Complete"<<std::endl;
}


void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
  
  // Define engine for random values
  std::default_random_engine gen;
  
  // Split the standard deviation from arguments
  double std_x, std_y, std_theta;
  std_x = std_pos[0];
  std_y = std_pos[1];
  std_theta = std_pos[2];
  
  // Loop through each particle
  for (int i=0; i<num_particles; ++i){
    
    // if the yaw rate is close to zero, use a different equation to avoid div by zero
    if (fabs(yaw_rate) > 0.00001){
      particles[i].x = particles[i].x + (velocity/yaw_rate)*(sin(particles[i].theta + (yaw_rate*delta_t)) - sin(particles[i].theta));
      particles[i].y = particles[i].y + (velocity/yaw_rate)*(cos(particles[i].theta) - cos(particles[i].theta + (yaw_rate* delta_t)));
      particles[i].theta = particles[i].theta + (yaw_rate* delta_t);
    }
    else{
      particles[i].x = particles[i].x + (velocity * delta_t * cos(particles[i].theta));
      particles[i].y = particles[i].y + (velocity * delta_t * sin(particles[i].theta));
      particles[i].theta = particles[i].theta;
    }
    
    // Add gaussian noise to the prediction
    normal_distribution<double> dist_x(particles[i].x, std_x);
    normal_distribution<double> dist_y(particles[i].y, std_y);
    normal_distribution<double> dist_theta(particles[i].theta, std_theta);
    
    // Update the particle values
    particles[i].x = dist_x(gen);
    particles[i].y = dist_y(gen);
    particles[i].theta = dist_theta(gen);
  }
  
  //cout<<"Prediction Complete"<<endl;
}




void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  /**
   * Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * This method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */
  
  // NOTE: This function is not used for this implementation
  double min_dist = 99999;
  double cur_dist = 0.0;
  int min_index = 0;
  
  // Loop through the obervation and find the correct closest landmark
  for (int j=0; j< observations.size(); ++j){
    for(int i=0; i<predicted.size(); ++i){
      cur_dist = dist(predicted[i].x, predicted[i].y, observations[j].x, observations[j].y);

      // Incomplete because not currently needed
      if(cur_dist < min_dist){
        min_dist = cur_dist;
        min_index = i;
      }
    }
  }
}



void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */
  double x_part, y_part, x_obs, y_obs, theta;
  double w_part;
  vector<long double> new_weights;
  
  // Vectors to hold the values for setting associations
  vector<int> associations;
  vector<double> sense_x;
  vector<double> sense_y;
  
  // Do the following for each particle:
    // Convert each vehicle observations to map coordinates using particle coordinates  
    // For each observation, get nearest neighbor from landmarks, using dist      
      // Use the nearest landmark, calculate the guassian prob for that observation
      // Final weight for that particle will be the product of all weights
  
  // Loop through each particle
  for(int i = 0; i<particles.size(); ++i){
    
    // Get the particle indices and theta
    x_part = particles[i].x;
    y_part = particles[i].y;
    theta = particles[i].theta;
    
    // Get landmark standard deviations
    double sig_x = std_landmark[0];
    double sig_y = std_landmark[1];
       
    long double weights_obs = 1.0;      // Stores the product of weights for all observations
    
    // Loop through all observations
    for (int j = 0; j< observations.size(); ++j){
      
      // Get observation indices
      x_obs = observations[j].x;
      y_obs = observations[j].y;

      // Transform to map coordinates
      double x_map;
      double y_map;
      x_map = x_part + (cos(theta) * x_obs) - (sin(theta) * y_obs);
      y_map = y_part + (sin(theta) * x_obs) + (cos(theta) * y_obs);
      
      // Find the nearest neighbor
      int nearest_lm_index = -1;    // Index of the nearest landmark
      int nearest_lm_id = -1;       // ID of the nearest landmark
      double min_dist = 99999.0;

      // Loop though all landmarks to find the closest one      
      for(int k=0; k< map_landmarks.landmark_list.size(); ++k){        
        
        // Calculate distance from landmark
        double cur_dist= 0.0;
        cur_dist = dist(map_landmarks.landmark_list[k].x_f, map_landmarks.landmark_list[k].y_f, x_map, y_map);
        
        // Store the closest one 
        if (cur_dist < min_dist){
          min_dist = cur_dist;
          nearest_lm_index = k;
          nearest_lm_id = map_landmarks.landmark_list[k].id_i;
        }        
      }
      
      // Calculate the probability distribution for the nearest neighbor
      // Use landmark indices as mean     
      double mu_x = map_landmarks.landmark_list[nearest_lm_index].x_f;
      double mu_y = map_landmarks.landmark_list[nearest_lm_index].y_f;

      long double gauss_norm = 0.0;
      long double exponent = 0.0;
      long double weight_cur_obs = 0.0;
      
      // Calculate the weight using the multivariate gaussian formula
      gauss_norm = 1.0 / (2.0 * M_PI * sig_x * sig_y);
      exponent = (pow(x_map - mu_x, 2.0) / (2.0 * pow(sig_x, 2.0))) + (pow(y_map - mu_y, 2.0) / (2.0 * pow(sig_y, 2.0)));
      weight_cur_obs = gauss_norm * exp(-exponent);   // calculate weight using normalization terms and exponent

      // Calculate Product of all observation weights
      weights_obs = weights_obs * weight_cur_obs;
      
      // Set association values
      associations.push_back(nearest_lm_id);
      sense_x.push_back(x_map);
      sense_y.push_back(y_map);
    }
    
    // Add the weight product for this particle to a vector
    new_weights.push_back(weights_obs);
    
    // Call function to set associations for the particle
    SetAssociations(particles[i], associations, sense_x, sense_y);    
  }
    
  // Normalize the weights

  // Get the sum of weights
  long double sum_weights = 0.0;    
  for(int m=0; m<num_particles; ++m){
    sum_weights += particles[m].weight;
  }

  //cout<<"sum_weights: "<<sum_weights<<endl;

  // Update the particle weight   
  for(int m=0; m<num_particles; ++m){
    particles[m].weight = particles[m].weight*(new_weights[m]/sum_weights);
  }
}



void ParticleFilter::resample() {
  /**
   * Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
  
  // Definitions for random uniform distribution
  std::random_device rd;  //Will be used to obtain a seed for the random number engine
  std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
  std::uniform_real_distribution<> dis(0.0, 1.0);
  
  std::vector<Particle> new_particles;   // Vector to hold the radom weighted selection
  
  double beta = 0.0;
  double max_wt = 0.0;
  int index = rand() % num_particles;
  
  // Find the max weight
  for(int i=0; i<num_particles; ++i){
      if(particles[i].weight > max_wt){
        max_wt = particles[i].weight;
      }
  }
  

  // Use a random beta value to sample with replacement
  // Probability of selection depeds on the particle weight
  // Using Beta sampling method explained by Sebastian
  for(int i=0; i<num_particles; ++i){
    beta = beta + (dis(gen)* max_wt*2.0);
    while(beta> particles[index].weight){
      beta = beta - particles[index].weight;
      index = (index+1)% num_particles;
    }
    // Add the particle to a temporary new vector    
    new_particles.push_back(particles[index]); 
  }
  
  // Update the set of particles based on above sampling selection
  particles = new_particles; 
}



void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}