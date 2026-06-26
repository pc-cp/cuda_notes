#include <gtest/gtest.h>

int main(int argc, char* argv[])
{
	::testing::InitGoogleTest(&argc, argv);
	// 0 if all the tests are successful, or 1 otherwise
	return RUN_ALL_TESTS();
}
