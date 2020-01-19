# Texture Synthesis for Irregularly Arranged Patterns
Given a small image (.jpg or .png) of a pattern that consists of some element repeating in a random or irregular manner, this program will attempt to create a larger image of specified size that follows the same pattern. The general algorithm used is a slightly modified version of the algorithm discussed in [this paper](https://people.eecs.berkeley.edu/~efros/research/quilting/quilting.pdf)
## General Description

Image Synthesis is done by capturing smaller square regions within the provided image. 

Each of these square regions are then divided into 2 parts. The central region and the periphery region.

First, one of these smaller squares is selected at random to be placed in the top left corner and one of the squares whose left periphery is deemed similar to the right periphery of the starting square is concatenated to the right of it. This process repeats until the first row is filled. For every subsequent row, similarity between the top periphery of the squares to be added and the bottom periphery of the already placed squares is also checked.

Finally, a dynamic programming-based approach using [seam carving](https://en.wikipedia.org/wiki/Seam_carving#Improvements_and_extensions) is done to find an optimal boundary between the left and right periphery to minimize the visual differences. In addition, a simple blur is also applied.


## Intended Use

In general, this program is intended to be used when the user has a somewhat small image of an irregular pattern that he/she needs to be larger (e.g. to use as a desktop/phone background or for a website). The quick alternative approach of repeatedly concatenating the image to itself until it reaches the desired size is generally insufficient since adding repetition typically fails to capture the randomness of the pattern in the original image. (Seen Below)

### Examples

