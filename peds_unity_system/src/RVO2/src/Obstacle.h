/*
 * Obstacle.h
 * RVO2 Library
 *
 * Copyright 2008 University of North Carolina at Chapel Hill
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Please send all bug reports to <geom@cs.unc.edu>.
 *
 * The authors may be contacted via:
 *
 * Jur van den Berg, Stephen J. Guy, Jamie Snape, Ming C. Lin, Dinesh Manocha
 * Dept. of Computer Science
 * 201 S. Columbia St.
 * Frederick P. Brooks, Jr. Computer Science Bldg.
 * Chapel Hill, N.C. 27599-3175
 * United States of America
 *
 * <http://gamma.cs.unc.edu/RVO2/>
 */

#ifndef RVO_OBSTACLE_H_
#define RVO_OBSTACLE_H_

/**
 * \file       Obstacle.h
 * \brief      Contains the Obstacle class.
 */

#include "Definitions.h"

namespace RVO {
	/**
	 * \brief      Defines static obstacles in the simulation.
	 */
	class Obstacle {
	private:
		/**
		 * \brief      Constructs a static obstacle instance.
		 */
		Obstacle();

		bool isConvex_;
		Obstacle *nextObstacle_;
		Vector2 point_;
		Obstacle *prevObstacle_;
		Vector2 unitDir_;

		size_t id_;

	public:
		enum NearTypeEnum {
				FIRST,
				MIDDLE,
				LAST
			};

		/*!
			 *	@brief		Computes the squared distance from the obstacle to the given point.
			 *				Also sets the value of the point in the provided Vector2
			 *
			 *	@param		pt			The point whose distance is to be evaluated
			 *	@param		nearPt		The position on the obstacle which is nearest to the
			 *							test point will be set here.
			 *	@param		distSq		The squared distance to the line (i.e. ||pt - nearPt||^2)
			 *							is placed inside this parameter.
			 *	@returns	The classificaiton of what the nearest point is - first, middle, or
			 *				last.
			 */
		NearTypeEnum distanceSqToPoint( const Vector2 & pt, Vector2 & nearPt,
											float & distSq ) const;

		friend class Agent;
		friend class KdTree;
		friend class RVOSimulator;
	};
}

#endif /* RVO_OBSTACLE_H_ */
