/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[])
{
	if (!is_initialized)
	{
		// set number of particles
		num_particles = 500;

		// Creates a normal (Gaussian) distribution
		normal_distribution<double> dist_x(x, std[0]);
		normal_distribution<double> dist_y(y, std[1]);
		normal_distribution<double> dist_theta(theta, std[2]);

		// Create particles and their weights based on the num_particles
		particles.resize(num_particles);
		weights.resize(num_particles);
		
		for (int i = 0; i < num_particles; i++)
		{
			Particle part;
			part.id     = i;
			part.x      = dist_x(gen);
			part.y      = dist_y(gen);
			part.theta  = dist_theta(gen);
			part.weight = 1.0;

			particles.push_back(part);
		}

		// set initialization to true
		is_initialized = true;
	}
	else
	{
		return;
	}
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate)
{
	// Creates a normal (Gaussian) distribution
	normal_distribution<double> dist_x(0.0, std_pos[0]);
	normal_distribution<double> dist_y(0.0, std_pos[1]);
	normal_distribution<double> dist_theta(0.0, std_pos[2]);

	for (double i = 0; i < particles.size(); i++)
	{
		double theta = particles[i].theta;

		// Case yaw_rate != 0
		if (fabs(yaw_rate) > 0.0001)
		{
			particles[i].x     += (velocity / yaw_rate) * (sin(theta + yaw_rate * delta_t) - sin(theta)) + dist_x(gen);
			particles[i].y     += (velocity / yaw_rate) * (cos(theta) - cos(theta + yaw_rate * delta_t)) + dist_y(gen);
			particles[i].theta += yaw_rate * delta_t + dist_theta(gen);
		}
		else // Case yaw_rate == 0
		{
			particles[i].x     += velocity * delta_t * cos(theta) + dist_x(gen);
			particles[i].y     += velocity * delta_t * sin(theta) + dist_y(gen);
			particles[i].theta += dist_theta(gen);
		}
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations)
{
	for (double i = 0; i < observations.size(); i++)
	{
		// set close point id to -1 (invalid id)
		double close_pt_id = -1;

		// set min. distance to the max. value
		double min_dist = numeric_limits<double>::max();

		for (double j = 0; j < predicted.size(); j++)
		{
			double distance = dist(observations[i].x, observations[i].y, predicted[j].x, predicted[j].y);

			if (distance < min_dist)
			{
				min_dist    = distance;
				close_pt_id = predicted[j].id;
			}
		}

		observations[i].id = close_pt_id;
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], const std::vector<LandmarkObs> &observations, const Map &map_landmarks)
{
	for (double i = 0; i < particles.size(); i++)
	{
		// 1. Locate all the landmarks inside sensor range
		vector<LandmarkObs> filtered_landmarks;
		for (double j = 0; j < map_landmarks.landmark_list.size(); j++)
		{
			double distance = dist(particles[i].x, particles[i].y, map_landmarks.landmark_list[j].x_f, map_landmarks.landmark_list[j].y_f);

			if (distance <= sensor_range)
			{
				LandmarkObs temp;
				temp.x  = map_landmarks.landmark_list[j].x_f;
				temp.y  = map_landmarks.landmark_list[j].y_f;
				temp.id = map_landmarks.landmark_list[j].id_i;

				filtered_landmarks.push_back(temp);
			}
		}

		// 2. Transform all the observations from vehicle coordinates to map coordinate
		vector<LandmarkObs> trans_obs;

		for (double j = 0; j < observations.size(); j++)
		{
			LandmarkObs temp;
			temp.x  = ( observations[j].x * cos(particles[i].theta) ) - ( observations[j].y * sin(particles[i].theta) ) + particles[i].x;
			temp.y  = ( observations[j].x * sin(particles[i].theta) ) + ( observations[j].y * cos(particles[i].theta) ) + particles[i].y ;
			temp.id = observations[j].id;

			trans_obs.push_back(temp);
		}
		
		// 3. Associate the filtered landmarks with the sensor range with trans_obs (observations into map coordinate)
		dataAssociation(filtered_landmarks, trans_obs);

		// 4. Update weights for each Particles
		for (double j = 0; j < trans_obs.size(); j++)
		{
			LandmarkObs chosen_part;

			for (unsigned k = 0; k < filtered_landmarks.size(); k++)
			{
				if (filtered_landmarks[k].id == trans_obs[j].id)
				{
					chosen_part.x = filtered_landmarks[k].x;
					chosen_part.y = filtered_landmarks[k].y;
				}

			}

			double x_norm = -pow(trans_obs[j].x - chosen_part.x, 2) / ( 2 * pow( std_landmark[0], 2) );
			double y_norm = -pow(trans_obs[j].y - chosen_part.y, 2) / ( 2 * pow( std_landmark[1], 2) );

			particles[i].weight = (1/(2*M_PI * std_landmark[0] * std_landmark[1])) * exp(x_norm + y_norm); 
		}

	}
}

void ParticleFilter::resample()
{
	vector<Particle> sampled_part;

	double max_weight = numeric_limits<double>::min();;

	for (int i = 0; i < num_particles; i++)
	{
		weights[i] = particles[i].weight;
		if (particles[i].weight > max_weight)
		{
			max_weight = particles[i].weight;
		}
	}

	uniform_real_distribution<double> beta_uniform(0.0, 2.0 * max_weight);
	uniform_int_distribution<int> index_uniform(0, num_particles - 1);

	auto index = index_uniform(gen);
	double beta = 0.0;
	for (int i = 0; i < num_particles; i++)
	{
		beta += beta_uniform(gen);
    	while (beta > weights[index])
		{
			beta -= weights[index];
        	index = (index + 1) % num_particles;
		}
		sampled_part.push_back(particles[index]); 
	}

    particles = sampled_part;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;

	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
