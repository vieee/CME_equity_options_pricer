"""
Test the fixed price fallback functionality
"""
import sys
sys.path.append('src')

def test_price_fallback():
    print("=== Testing Price Fallback Functionality ===")
    
    try:
        from utils.cache import cached_stock_info
        
        # Test with SPY
        print("Testing SPY stock info with price fallback...")
        stock_info = cached_stock_info('SPY')
        
        if stock_info:
            current_price = stock_info['currentPrice']
            previous_close = stock_info['previousClose']
            
            print(f"✅ Current Price: ${current_price:.2f}")
            print(f"✅ Previous Close: ${previous_close:.2f}")
            
            if current_price > 0:
                print("✅ SUCCESS! Price is not zero")
                
                if abs(current_price - previous_close) < 0.01 and previous_close > 0:
                    print("📝 Note: Current price equals previous close (likely using fallback)")
                else:
                    print("📈 Note: Current price differs from previous close (likely live data)")
                    
            else:
                print("❌ Price is still zero after fallback")
                
            return current_price > 0
        else:
            print("❌ Failed to get stock info")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_multiple_symbols():
    print("\n=== Testing Multiple Symbols ===")
    
    symbols = ['SPY', 'AAPL', 'QQQ', 'MSFT', 'GOOGL']
    
    try:
        from utils.cache import cached_stock_info
        
        for symbol in symbols:
            print(f"\nTesting {symbol}...")
            stock_info = cached_stock_info(symbol)
            
            if stock_info:
                current_price = stock_info['currentPrice']
                previous_close = stock_info['previousClose']
                
                if current_price > 0:
                    print(f"✅ {symbol}: ${current_price:.2f}")
                else:
                    print(f"❌ {symbol}: Price is zero")
            else:
                print(f"❌ {symbol}: Failed to get data")
                
    except Exception as e:
        print(f"❌ Error testing multiple symbols: {e}")

def test_options_data_price():
    print("\n=== Testing Options Data Price Fallback ===")
    
    try:
        from utils.cache import cached_options_data
        
        print("Testing SPY options data with price fallback...")
        options_df, current_price = cached_options_data('SPY')
        
        if options_df is not None and current_price is not None:
            print(f"✅ Options data retrieved successfully")
            print(f"✅ Current price from options: ${current_price:.2f}")
            
            if current_price > 0:
                print("✅ SUCCESS! Options current price is not zero")
                return True
            else:
                print("❌ Options current price is still zero")
                return False
        else:
            print("❌ Failed to get options data")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    test1 = test_price_fallback()
    test_multiple_symbols()
    test2 = test_options_data_price()
    
    if test1 and test2:
        print("\n🎉 All price fallback tests passed!")
        print("The application should now show proper prices instead of $0.00")
    else:
        print("\n❌ Some tests failed. Please check the output above.")
