# to save results
import datetime
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

from stock_analyzer import comprehensive_stock_analysis, simple_test


def test_single_stock():
    """Test with a single stock first"""
    print("🚀 Testing Fixed Version")
    print("=" * 60)

    # First run the simple test
    simple_test()

    print("\n" + "=" * 60)
    print("Now testing comprehensive analysis...")
    print("=" * 60)

    # Test with Aramco
    try:
        print("\n📊 Testing comprehensive analysis for Aramco...")
        data, metrics = comprehensive_stock_analysis(
            ticker_symbol='2222.SR',
            ticker_name='Saudi Aramco',
            period='3mo'  # Use 3 months for testing
        )

        if data is not None and metrics is not None:
            print("\n✅ ANALYSIS SUCCESSFUL!")

            # Save results
            import pandas as pd

            # Save price data
            data.to_csv('aramco_analysis_results.csv')
            print("💾 Price data saved to: aramco_analysis_results.csv")

            # Save metrics
            metrics_df = pd.DataFrame([metrics])
            metrics_df.to_csv('aramco_metrics.csv', index=False)
            print("💾 Metrics saved to: aramco_metrics.csv")

            # Show summary
            print("\n📈 Summary Metrics:")
            for key, value in metrics.items():
                if value is not None:
                    print(f"  {key}: {value}")

    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main function"""
    print("🚀 Tadawul Stock Analysis Tool - Fixed Version")
    print("=" * 60)

    # List of Saudi stocks
    saudi_stocks = {
        'Aramco': '2222.SR',
        'SABIC': '2010.SR',
        'STC': '7010.SR',
        'Al Rajhi Bank': '1120.SR'
    }

    print("\n📋 Available Saudi Stocks:")
    print("-" * 30)
    for idx, (name, symbol) in enumerate(saudi_stocks.items(), 1):
        print(f"{idx}. {name} ({symbol})")

    # Get user choice
    try:
        choice = input("\nSelect stock number (1-4): ").strip()

        if choice.isdigit() and 1 <= int(choice) <= 4:
            stock_list = list(saudi_stocks.items())
            stock_name, stock_symbol = stock_list[int(choice) - 1]

            # Run analysis
            print(f"\n{'=' * 60}")
            print(f"Starting analysis for {stock_name}...")
            print(f"{'=' * 60}")

            data, metrics = comprehensive_stock_analysis(
                ticker_symbol=stock_symbol,
                ticker_name=stock_name,
                period='3mo'  # Default to 3 months
            )

            if data is not None and metrics is not None:
                print("\n✅ Analysis completed successfully!")
        else:
            print("❌ Invalid choice. Please select 1-4.")

    except KeyboardInterrupt:
        print("\n\n⚠️ Analysis interrupted by user.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 60)
    print("✅ Program finished!")
    print("=" * 60)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == '--test':
        test_single_stock()
    else:
        main()
