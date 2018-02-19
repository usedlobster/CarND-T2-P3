/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */


// completed by usedlobster Feburary 2018.

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>
#include <assert.h>
#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {

    // Set the number of initial particles, to use

    // anything > 10 works , 200 seems a good trade off between time / accuracy

    num_particles = 200 ;


    //Initialize all particles to first position (based on estimates of
    // x, y, theta and their uncertainties from GPS) and all weights to 1.
    // Add random Gaussian noise to each particle.

    // create a ( pseudo ) random number engine
    default_random_engine generator ;

    // create gaussian distribution generators for each standard deviation.
    normal_distribution<double> dist_x( x, std[0] ) ;
    normal_distribution<double> dist_y( y, std[1] ) ;
    normal_distribution<double> dist_theta( theta, std[2] );

    // reserve space for efficiency ,
    particles.reserve( num_particles ) ;

    // create a random particle - using generators above.

    for ( int i=0; i<num_particles; i++) {

        Particle a_random_particle ;

        a_random_particle.id    = i                 ;
        a_random_particle.x     = dist_x( generator )     ;
        a_random_particle.y     = dist_y( generator )     ;
        a_random_particle.theta = dist_theta( generator ) ;

        particles.push_back( a_random_particle ) ;

    }
    // need todo this
    is_initialized = true ;
}



void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {


    default_random_engine generator ;

    // create normal distribution generator with 0 mean , std_pos
    normal_distribution<double> dist_x( 0.0, std_pos[0] );
    normal_distribution<double> dist_y( 0.0, std_pos[1] );
    normal_distribution<double> dist_theta( 0.0, std_pos[2] );


    // apply velocity , and yaw change to each particle

    for ( auto &p : particles ) {

        // if yaw_rate is practically 0
        // just move forward in current direction.
        if ( fabs( yaw_rate) < 1e-5 ) {
            p.x += velocity * delta_t * cos( p.theta ) ;
            p.y += velocity * delta_t * sin( p.theta ) ;
        } else {
            p.x += ( velocity / yaw_rate ) * ( sin( p.theta + yaw_rate * delta_t )  - sin( p.theta )) ;
            p.y += ( velocity / yaw_rate ) * ( cos( p.theta ) - cos( p.theta + yaw_rate * delta_t )) ;
            p.theta += yaw_rate*delta_t ;
        }

        // add some noise to the position

        p.x += dist_x( generator ) ;
        p.y += dist_y( generator ) ;
        p.theta += dist_theta( generator ) ;

    }

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations)
{

    // go through each observation , and find which landmark is nearest.

;    for ( auto &obs : observations ) {

        double min_d2       ; // NB: no need to initialize
        int nearest_id = -1 ;


        for ( auto lm : predicted ) {
            //
            double d2 = ( obs.x - lm.x )*( obs.x - lm.x ) + ( obs.y - lm.y )*( obs.y - lm.y ) ;

            if ( nearest_id < 0 || d2 < min_d2 ) {
                nearest_id = lm.id ;
                min_d2 = d2        ;
            }
        }

        // record  which landmark this observation was closest to.
        obs.id = nearest_id ;
    }




}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {

    //  Update the weights of each particle using a mult-variate Gaussian distribution.


    // precompute these, as used for each particle.

    double sigX   = std_landmark[0] ; // sigma_x
    double sigY   = std_landmark[1] ; // sigma_y
    double gNorm  = 1.0 / ( 2.0 * M_PI * sigX * sigY ) ; // 1/( 2*pi*sigma_x*sigma_y )
    double sig2XX = 2.0 * sigX * sigX ; // twice sigma_x squared
    double sig2YY = 2.0 * sigY * sigY ; // twice sigma_y squared

    // clear vector list of weights
    weights.clear() ;

    // for each particle we
    for ( auto &p : particles ) {


        // create a list of observation in real world space
        // as if we had observed them from current particle p .

        std::vector< LandmarkObs >real_world_obs ;


        double cos_t = cos( p.theta ) ;
        double sin_t = sin( p.theta ) ;


        for ( auto obs : observations ) {


            double mx = p.x + obs.x * cos_t - obs.y * sin_t ;
            double my = p.y + obs.x * sin_t + obs.y * cos_t ;

            real_world_obs.push_back( LandmarkObs{ obs.id, mx, my } ) ;

        }

        // create list of possible landmarks -  in sensor range ( a square area  )

        std::vector< LandmarkObs >nearby_landmarks ;
        for  ( auto lm : map_landmarks.landmark_list  ) {
            if ( fabs( lm.x_f - p.x  ) < sensor_range || fabs( lm.y_f - p.y ) < sensor_range )
                nearby_landmarks.push_back( LandmarkObs{ lm.id_i, lm.x_f, lm.y_f } ) ;
        }

        // associate the real_world_obs with possible landmarks.
        dataAssociation( nearby_landmarks, real_world_obs ) ;

        // we can calculate the weight of each particle

        p.associations.clear() ;
        p.sense_x.clear() ;
        p.sense_y.clear() ;

        if ( nearby_landmarks.size() > 0 ) {

            p.weight = 1.0 ;

            for ( auto rwo : real_world_obs ) {

                if ( rwo.id >= 0 )  {

                    // rwo.id gives us best landmark to use , its -1 if there is none
                    // we need to iterate to find it again,
                    // as we haven't assumed map_landmarks.landmark_lists[rwo.id-1] is the same landmark
                    // but it probably is. ( acctually is ).
                    // We have also assumed id's are always >=0

                    for ( int i=0; i < nearby_landmarks.size() ; i++ ) {
                        if ( nearby_landmarks[i].id == rwo.id ) {
                            double dx = rwo.x - nearby_landmarks[i].x ;
                            double dy = rwo.y - nearby_landmarks[i].y ;
                            //
                            p.weight *= gNorm * exp( - ( (( dx * dx ) / sig2XX ) + (( dy * dy ) / sig2YY ))) ;


                            // we might as well assign debug information here
                            // rather than create 3 more lists.

                            p.associations.push_back( rwo.id ) ;
                            p.sense_x.push_back( rwo.x ) ;
                            p.sense_y.push_back( rwo.y ) ;

                            break ;
                        }
                    }
                }
            }
        } else
            p.weight = 0.0 ; // set to 0 as no landmarks

        // keep track of particle weights in a single vector list as well.
        // helps with resampling
        weights.push_back( p.weight ) ;
    }
}

void ParticleFilter::resample() {


    default_random_engine gen;
    std::discrete_distribution<int> weight_distribution( weights.begin(), weights.end());

    std::vector<Particle> resampled_particles ;

    resampled_particles.clear() ; // not strictly necessary I know.

    for (int i = 0; i < particles.size() ; i++)
        resampled_particles.push_back( particles[weight_distribution(gen)] );

    particles = resampled_particles ;


// I prefer old wheel method - works just as well , and you can see what its doing
    /*

        std::uniform_real_distribution<double> uniR(0.0, 1.0 );


        double wMax = *std::max_element( weights.begin(), weights.end());

        int N = weights.size() ;

        double beta = 0.0 ;
        int index = int(N*uniR(gen))%N ;
        for ( int i=0; i<N; i++) {
            beta = beta +  2.0 * uniR(gen) * wMax ;
            while ( weights[index] < beta ) {
                beta = beta - weights[index] ;
                index = (index+1)%N ;
            }

            resampled_particles.push_back( particles[index] );
        }

        particles = resampled_particles ;
    */



}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations,
        const std::vector<double>& sense_x, const std::vector<double>& sense_y) {
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
    vector<int> v = best.associations;
    stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best) {
    vector<double> v = best.sense_x;
    stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best) {
    vector<double> v = best.sense_y;
    stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
